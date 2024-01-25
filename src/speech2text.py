import argparse
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
from pydub import AudioSegment

import torch.multiprocessing as mp

import numpy as np
from whisperx import load_audio, load_model
from whisperx.asr import WhisperModel
from whisperx.types import TranscriptionResult

import pandas as pd
import torch
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)

from submit import parse_output_dir
from utils import DiarizationPipeline, seconds_to_human_readable_format

import settings

# https://numba.pydata.org/numba-doc/dev/reference/deprecation.html
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("__name__")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Sharing CUDA tensors between prcoesses requires a spawn or forkserver start method
if __name__=='__main__':
    mp.set_start_method('spawn')

def get_argument_parser():
    parser = argparse.ArgumentParser(
        prog="Aalto speech2text",
        description="This script does speech-to-text with diarization using OpenAI Whisper",
    )
    parser.add_argument(
        "INPUT_FILE",
        type=str,
        help="Input audio file OR a text file with a single input audio file on every line. Mandatory.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_TMP",
        type=str,
        default=os.getenv("SPEECH2TEXT_TMP"),
        help="Temporary folder. If not given, should be set as an environment variable.",
    )
    parser.add_argument(
        "--AUTH_TOKEN",
        type=str,
        default=os.getenv("AUTH_TOKEN"),
        help="Either AUTH_TOKEN or PYANNOTE_CONFIG environment variable needs to be set.",
    )
    parser.add_argument(
        "--PYANNOTE_CONFIG",
        type=str,
        default=os.getenv("PYANNOTE_CONFIG"),
        help="Either AUTH_TOKEN or PYANNOTE_CONFIG environment variable needs to be set.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_LANGUAGE",
        type=str,
        default=os.getenv("SPEECH2TEXT_LANGUAGE"),
        help="Audio language. Optional but recommended.",
    )

    return parser


def compute_overlap(start1, end1, start2, end2):
    if start1 > end1 or start2 > end2:
        raise ValueError("Start of segment can't be larger than its end.")

    start_overlap = max(start1, start2)
    end_overlap = min(end1, end2)

    if start_overlap > end_overlap:
        return 0

    return abs(end_overlap - start_overlap)


def align(segments, diarization):
    """
    Align diarization with transcription.

    Transcription and diarization segments is measured using overlap in time.

    If no diarization segment overlaps with a given transcription segment, the speaker
    for that transcription segment is None.

    The output object is a dict of lists:

    {
    "start" : [0.0, 4.5, 7.0],
    "end"   : [3.3, 6.0, 10.0],
    "transcription" : ["This is first first speaker segment", "This is the second", "This is from an unknown speaker"],
    "speaker": ["SPEAKER_00", "SPEAKER_01", None]
    }

    Parameters
    ----------
    transcription : list
        Output of Whisper transcribe()
    diarization : list
        Output of Pyannote diarization()

    Returns
    -------
    dict
    """
    transcription_segments = [
        (segment['start'], segment['end'], segment['text']) for segment in segments
    ]
    diarization_segments = [
        (start, end, speaker)
        for _, _, speaker, start, end in diarization.to_numpy()
    ]
    alignment = defaultdict(list)
    for transcription_start, transcription_end, text in transcription_segments:
        max_overlap, max_speaker = None, None
        for diarization_start, diarization_end, speaker in diarization_segments:
            overlap = compute_overlap(
                transcription_start,
                transcription_end,
                diarization_start,
                diarization_end,
            )
            if overlap > 0 and (max_overlap is None or overlap > max_overlap):
                max_overlap, max_speaker = overlap, speaker

        transcription_start = seconds_to_human_readable_format(transcription_start)
        transcription_end = seconds_to_human_readable_format(transcription_end)

        alignment["start"].append(transcription_start)
        alignment["end"].append(transcription_end)
        alignment["speaker"].append(max_speaker)
        alignment["transcription"].append(text.strip())

    return alignment


def parse_output_file_stem(output_dir, input_file):
    return Path(output_dir) / Path(Path(input_file).name)


def write_alignment_to_csv_file(alignment, output_file_stem):
    df = pd.DataFrame.from_dict(alignment)
    output_file = str(Path(output_file_stem).with_suffix(".csv"))
    df.to_csv(
        output_file,
        sep=",",
        index=False,
        encoding="utf-8",
    )
    logger.info(f".. .. Wrote CSV output to: {output_file}")


def write_alignment_to_txt_file(alignment, output_file_stem):
    # Group lines by speaker
    all_lines_grouped_by_speaker = []
    lines_speaker = []
    prev_speaker = None
    for start, end, speaker, transcription in zip(
        alignment["start"],
        alignment["end"],
        alignment["speaker"],
        alignment["transcription"],
    ):
        if speaker != prev_speaker and lines_speaker:
            all_lines_grouped_by_speaker.append(lines_speaker)
            lines_speaker = []
        lines_speaker.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "transcription": transcription,
            }
        )
        prev_speaker = speaker

    # Append remainders
    if lines_speaker:
        all_lines_grouped_by_speaker.append(lines_speaker)

    # Write out
    lines_out = []
    for lines_speaker in all_lines_grouped_by_speaker:
        start = lines_speaker[0]["start"]  # first start time in group
        end = lines_speaker[-1]["end"]  # last end time in group
        speaker = lines_speaker[0]["speaker"]  # same speaker across group
        meta_info_line = f"({start} - {end}) {speaker}\n\n"
        lines_out.append(meta_info_line)
        transcriptions = (
            " ".join([line["transcription"] for line in lines_speaker]) + "\n\n"
        )
        lines_out.append(transcriptions)

    output_file = str(Path(output_file_stem).with_suffix(".txt"))
    with open(output_file, "w") as fout:
        for line in lines_out:
            fout.write(line)

    logger.info(f".. .. Wrote TXT output to: {output_file}")


def load_whisperx_model(
    name: str = "large-v3",
    device: Optional[Union[str, torch.device]] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        name,
        device=device,
        threads=6,
        compute_type="float16",
    )
    return model


def read_input_file_from_array_file(input_file, slurm_array_task_id):
    logger.info(f".. Read item {slurm_array_task_id} from {input_file}")
    input_files = []
    with open(input_file, "r") as fin:
        input_files = json.load(fin)
    logger.info(f".. .. Read items: {input_files}")
    new_input_file = input_files[int(slurm_array_task_id)]
    logger.info(f".. Return: {new_input_file}")
    return new_input_file


def transcribe(file: str, language: str, result_list) -> TranscriptionResult:
    
    model = load_whisperx_model()
    segs, lang = model.transcribe(file, batch_size=32, language=language).values()
    
    result_list['segments'] = segs
    result_list['transcribe_time'] = time.time()


def diarization(file: str, config: str, token: str, result_list):

    diarization_pipeline = DiarizationPipeline(config_file=config, 
                                               auth_token=token)
    diarization = diarization_pipeline(file)
    
    result_list['diarization'] = diarization
    result_list['diarize_time'] = time.time()


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    logger.info(f"Start processing input file: {args.INPUT_FILE}")

    if not Path(args.INPUT_FILE).is_file():
        logger.error(f".. Given input file '{args.INPUT_FILE}' does not exist!")
        return

    # Parse input file
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if Path(args.INPUT_FILE).suffix == ".json" and slurm_array_task_id is not None:
        args.INPUT_FILE = read_input_file_from_array_file(
            args.INPUT_FILE, slurm_array_task_id
        )

    # .wav conversion
    logger.info(f".. Convert input file to wav format for pyannote diarization pipeline: {args.INPUT_FILE}")
    t0 = time.time()
    try:
        input_file_wav = load_audio(args.INPUT_FILE)
    except Exception as e:
        logger.error(f".. .. Input file could not be converted: {args.INPUT_FILE}")
        raise(e)
        
    logger.info(f".. .. Wav conversion done in {time.time()-t0:.1f} seconds")
    
    language = args.SPEECH2TEXT_LANGUAGE
    if language and language.lower() in settings.supported_languages:
        language = settings.supported_languages[language.lower()]

    with mp.Manager() as manager:
        shared_dict = manager.dict()

        process1 = mp.Process(target=transcribe, args=(input_file_wav, language, shared_dict,))
        process2 = mp.Process(target=diarization, args=(args.INPUT_FILE, args.PYANNOTE_CONFIG, args.AUTH_TOKEN, shared_dict,))
        
        t0=time.time()
        logger.info(f".. Starting transcription task for {args.INPUT_FILE}")
        process1.start()

        logger.info(f".. Starting diarization task for {args.INPUT_FILE}")
        process2.start()

        process1.join()
        process2.join()
        
        logger.info(f".. .. Transcription finished in {shared_dict['transcribe_time']-t0:.1f} seconds")
        logger.info(f".. .. Diarization finished in {shared_dict['transcribe_time']-t0:.1f} seconds")

        segments = shared_dict['segments']
        diarization_results = shared_dict['diarization']
    
    # Alignment
    logger.info(".. Align transcription and diarization")
    alignment = align(segments, diarization_results)

    logger.info(f".. Write alignment to output")
    output_dir = parse_output_dir(args.INPUT_FILE)
    output_file_stem = parse_output_file_stem(output_dir, args.INPUT_FILE)
    write_alignment_to_csv_file(alignment, output_file_stem)
    write_alignment_to_txt_file(alignment, output_file_stem)

    logger.info(f"Finished.")


if __name__ == "__main__":
    main()
