import argparse
import gc
import json
import logging
import os
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.multiprocessing as mp
import whisperx
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from whisperx.types import TranscriptionResult

from .settings import (available_whisper_models,
                       wav2vec_models,
                       supported_languages_pretty,
                       DEFAULT_WHISPER_MODEL, 
                       DEFAULT_COMPUTE_DEVICE)
from .submit import parse_output_dir
from .utils import (DiarizationPipeline, 
                    assign_word_speakers,
                    calculate_max_batch_size,
                    convert_language_to_abbreviated_form, 
                    load_audio,
                    seconds_to_human_readable_format)
from .writer import (write_result_to_csv_file,
                     write_result_to_txt_file)

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
if __name__ == "__main__":
    mp.set_start_method("spawn")


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
        help="Audio language. Required; otherwise will raise an error.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_WHISPER_MODEL",
        type=str,
        default=os.getenv("SPEECH2TEXT_WHISPER_MODEL"),
        choices=available_whisper_models,
        help=f"Whisper model. Defaults to {DEFAULT_WHISPER_MODEL}.",
    )

    return parser


def combine_transcription_and_diarization(
    transcription_segments, diarization_segments, file, language
) -> dict:
    """
    Combine transcription and diarization results:

    1. Convert transcription segments using wav2vec alignment so that each segment corresponds to a word
    2. For each transcribed word, find the most overlapping (in time) diarization/speaker segment

    If no diarization segment overlaps with a word, the speaker for that word is "SPEAKER_UNKNOWN".

    Parameters
    ----------
    transcription_segments : list
        Output of Whisper transcribe()
    diarization_segments : list
        Output of Pyannote diarization()
    file : str
        The input audio file
    language: str
        language in short format (e.g. "fi")

    Returns
    -------
    dict:
        The output object is a dict of lists:
        {
        "start" : [0.0, 4.5, 7.0],
        "end"   : [3.3, 6.0, 10.0],
        "transcription" : ["This is first first speaker segment", "This is the second", "This is from an unknown speaker"],
        "speaker": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_UNKNOWN"]
        }
    """

    # Convert transcription segments so that each segment corresponds to a word
    wav2vec_model_name = (
        wav2vec_models[language]
        if language in wav2vec_models
        else None
    )

    align_model, align_metadata = whisperx.load_align_model(
        language, DEFAULT_COMPUTE_DEVICE, model_name=wav2vec_model_name
    )

    transcription_segments = whisperx.align(
        transcription_segments,
        align_model,
        align_metadata,
        file,
        DEFAULT_COMPUTE_DEVICE,
    )

    # Assign speaker to transcribed word segments
    segments = assign_word_speakers(
        diarization_segments, transcription_segments["segments"]
    )

    # Reformat the result (return a dictionary of lists)
    result = defaultdict(list)
    for segment in segments:
        transcription_start = seconds_to_human_readable_format(segment["start"])
        transcription_end = seconds_to_human_readable_format(segment["end"])

        result["start"].append(transcription_start)
        result["end"].append(transcription_end)
        result["transcription"].append(segment["text"].strip())
        try:
            result["speaker"].append(segment["speaker"])
        except KeyError:
            result["speaker"].append("SPEAKER_UNKNOWN")

    return result, time.time()


def parse_output_file_stem(output_dir: str, input_file: str) -> Path:
    """
    Create output file stem from output directory and input file name.
    """
    return Path(output_dir) / Path(Path(input_file).name)


def load_whisperx_model(
    name: str,
    language: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = DEFAULT_COMPUTE_DEVICE,
):
    """
    Load a Whisper model on GPU.

    Will raise an error if CUDA is not available. This is due to batch_size optimization method in utils.py.
    The submitted script will run on a GPU node, so this should not be a problem. The only issue is with a
    hardware failure.
    """
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is not available. Check the hardware failures for "
            + subprocess.check_output(["hostname"]).decode()
        )

    if name not in available_whisper_models:
        logger.warning(
            f"Specified model '{name}' not among available models: {available_whisper_models}. Opting to use the default model '{DEFAULT_WHISPER_MODEL}' instead"
        )
        name = DEFAULT_WHISPER_MODEL

    compute_type = "float16"
    try:
        model = whisperx.load_model(
            name,
            language=language,
            device=device,
            threads=6,
            compute_type=compute_type,
        )
    except ValueError:
        compute_type = "float32"
        model = whisperx.load_model(
            name,
            language=language,
            device=device,
            threads=6,
            compute_type=compute_type,
        )
    return model


def read_input_file_from_array_file(input_file: str, slurm_array_task_id: str):
    """
    Read a single audio path from a JSON file with an array of audio paths.

    Returns the audio path at the given index.
    """
    logger.info(f".. Read item {slurm_array_task_id} from {input_file}")
    input_files = []
    with open(input_file, "r") as fin:
        input_files = json.load(fin)
    logger.info(f".. .. Read items: {input_files}")
    new_input_file = input_files[int(slurm_array_task_id)]
    logger.info(f".. Return: {new_input_file}")
    return new_input_file


def transcribe(
    file: str, model_name: str, language: str, result: dict
) -> TranscriptionResult:
    """
    Transcribe audio file using WhisperX.

    Parameters
    ----------
    file : str
        The input audio file.
    model_name : str
        The Whisper model name.
    language : str
        The language of the audio. Not setting the language will result in automatic language detection.
    result : dict
        The dictionary to store the result.
    """
    batch_size = calculate_max_batch_size()
    model = load_whisperx_model(model_name, language)

    try:
        whisperx_result = model.transcribe(
            file, batch_size=batch_size, language=language
        )
        segments = whisperx_result["segments"]

    # If the batch size is too large, reduce it by half and try again to avoid CUDA memory error.
    except torch.cuda.CudaError:
        logger.warning(
            f"Current CUDA device {torch.cuda.current_device()} doesn't have enough memory. Reducing batch_size {batch_size} by half."
        )

        gc.collect()
        torch.cuda.empty_cache()

        batch_size /= 2
        whisperx_result = model.transcribe(
            file, batch_size=int(batch_size), language=language
        )
        segments = whisperx_result["segments"]

    result["transcription_segments"] = segments
    result["transcription_done_time"] = time.time()


def diarize(file: str, config: str, token: str, result_list: dict):
    """
    Diarize audio file using Pyannote.

    Parameters
    ----------
    file : str
        The input audio file.
    config : str
        Configuration for the Pyannote model.
    token : str
        Access token for the the Hugging Face model if the config file is not available.
    result_list : dict
        The dictionary to store the result.
    """
    diarization_pipeline = DiarizationPipeline(config_file=config, auth_token=token)
    diarization_segments = diarization_pipeline(file)

    result_list["diarization_segments"] = diarization_segments
    result_list["diarization_done_time"] = time.time()


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

    # Check mandatory language argument
    language = args.SPEECH2TEXT_LANGUAGE
    language = convert_language_to_abbreviated_form(language)
    if not language:
        error_message = f"\
            Language not given or not supported. Supported languages:\
            {supported_languages_pretty}"
        logger.error(error_message)
        raise ValueError(error_message)

    # .wav conversion
    logger.info(
        f".. Convert input file to wav format for pyannote diarization pipeline: {args.INPUT_FILE}"
    )
    t0 = time.time()
    try:
        input_file_wav, _, _ = load_audio(args.INPUT_FILE)
    except Exception as e:
        logger.error(f".. .. Input file could not be converted: {args.INPUT_FILE}")
        raise (e)

    logger.info(f".. .. Wav conversion done in {time.time()-t0:.1f} seconds")

    # Check Whisper model name if given
    model_name = args.SPEECH2TEXT_WHISPER_MODEL
    if model_name is None:
        logger.info(
            f"Whisper model not given. Opting to use the default model '{DEFAULT_WHISPER_MODEL}'"
        )
        model_name = DEFAULT_WHISPER_MODEL
    elif model_name not in available_whisper_models:
        logger.warning(
            f"Given Whisper model '{model_name}' not among available models: {available_whisper_models}. Opting to use the default model '{DEFAULT_WHISPER_MODEL}' instead"
        )
        model_name = DEFAULT_WHISPER_MODEL

    # Creating two separate processes for transcription and diarization based on torch multiprocessing
    with mp.Manager() as manager:
        shared_dict = manager.dict()

        process1 = mp.Process(
            target=transcribe,
            args=(
                input_file_wav,
                model_name,
                language,
                shared_dict,
            ),
        )
        process2 = mp.Process(
            target=diarize,
            args=(
                input_file_wav,
                args.PYANNOTE_CONFIG,
                args.AUTH_TOKEN,
                shared_dict,
            ),
        )

        t0 = time.time()
        logger.info(f".. Starting transcription task for {args.INPUT_FILE}")
        process1.start()

        logger.info(f".. Starting diarization task for {args.INPUT_FILE}")
        process2.start()

        process1.join()
        process2.join()

        logger.info(
            f".. .. Transcription finished in {shared_dict['transcription_done_time']-t0:.1f} seconds"
        )
        logger.info(
            f".. .. Diarization finished in {shared_dict['diarization_done_time']-t0:.1f} seconds"
        )

        transcription_segments = shared_dict["transcription_segments"]
        diarization_segments = shared_dict["diarization_segments"]
        if not language:
            # Grab the detected language from the transcription result
            logging.info(f".. Language not given. Detected language: {language}")
            language = shared_dict["language"]

        torch.cuda.empty_cache()

    t0 = time.time()
    logger.info(".. Combine transcription and diarization segments")
    combination, combination_done_time = combine_transcription_and_diarization(
        transcription_segments, diarization_segments, input_file_wav, language
    )
    logger.info(f".. .. Combining finished in {combination_done_time-t0:.1f} seconds")

    logger.info(f".. Write final result to files")
    output_dir = parse_output_dir(args.INPUT_FILE)
    output_file_stem = parse_output_file_stem(output_dir, args.INPUT_FILE)
    write_result_to_csv_file(combination, output_file_stem)
    write_result_to_txt_file(combination, output_file_stem)

    logger.info(f"Finished.")


if __name__ == "__main__":
    main()
