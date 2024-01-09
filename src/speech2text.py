import argparse
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from typing import Optional, Union
import pandas as pd
import faster_whisper
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from pyannote.audio import Pipeline

from submit import parse_output_dir
from utils import seconds_to_human_readable_format

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
        (segment.start, segment.end, segment.text)
        for segment in segments
    ]
    diarization_segments = [
        (segment.start, segment.end, speaker)
        for segment, _, speaker in diarization.itertracks(yield_label=True)
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


def load_whisper_model(name: str = "large-v3",
                       device: Optional[Union[str, torch.device]] = None,
                       ):    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = faster_whisper.WhisperModel(name, 
                                        device=device,
                                        cpu_threads=6,
                                        compute_type="int8",
                                        )

    return model


def load_pipeline(config_file, auth_token):
    """
    For more info on the config file, see 'Offline use' at:
    https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb
    """

    if Path(config_file).is_file():
        logger.info(".. .. Local config file found")
        pipeline = Pipeline.from_pretrained(config_file)
    elif auth_token:
        logger.info(".. .. Environment variable AUTH_TOKEN found")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token,
        )
    else:
        logger.error(
            "One of these is required: local pyannote config file or environment variable AUTH_TOKEN to download model from HuggingFace hub"
        )
        raise ValueError

    return pipeline


def read_input_file_from_array_file(input_file, slurm_array_task_id):
    logger.info(f".. Read item {slurm_array_task_id} from {input_file}")
    input_files = []
    with open(input_file, "r") as fin:
        input_files = json.load(fin)
    logger.info(f".. .. Read items: {input_files}")
    new_input_file = input_files[int(slurm_array_task_id)]
    logger.info(f".. Return: {new_input_file}")
    return new_input_file


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    logger.info(f"Start processing input file: {args.INPUT_FILE}")

    if not Path(args.INPUT_FILE).is_file():
        logger.error(f".. Given input file '{args.INPUT_FILE}' does not exist!")
        return

    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if Path(args.INPUT_FILE).suffix == ".json" and slurm_array_task_id is not None:
        args.INPUT_FILE = read_input_file_from_array_file(
            args.INPUT_FILE, slurm_array_task_id
        )

    logger.info(".. Load models")
    logging.info(args.PYANNOTE_CONFIG)
    diarization_pipeline = load_pipeline(args.PYANNOTE_CONFIG, args.AUTH_TOKEN)
    t0 = time.time()
    faster_whisper_model = load_whisper_model()
    logger.info(f".. .. Models loaded in {time.time()-t0:.1f} seconds")

    logger.info(f".. Transcribe input file: {args.INPUT_FILE}")
    t0 = time.time()
    segments, _ = faster_whisper_model.transcribe(args.INPUT_FILE,
                                                  language=args.SPEECH2TEXT_LANGUAGE,
                                                  beam_size=5)
    segments = list(segments)
    logger.info(f".. .. Transcription finished in {time.time()-t0:.1f} seconds")

    logger.info(f".. Diarize input file: {args.INPUT_FILE}")
    t0 = time.time()
    diarization = diarization_pipeline(args.INPUT_FILE)
    logger.info(f".. .. Diarization finished in {time.time()-t0:.1f} seconds")

    logger.info(".. Align transcription and diarization")
    alignment = align(segments, diarization)

    logger.info(f".. Write alignment to output")
    output_dir = parse_output_dir(args.INPUT_FILE)
    output_file_stem = parse_output_file_stem(output_dir, args.INPUT_FILE)
    write_alignment_to_csv_file(alignment, output_file_stem)
    write_alignment_to_txt_file(alignment, output_file_stem)

    logger.info(f"Finished.")


if __name__ == "__main__":
    main()
