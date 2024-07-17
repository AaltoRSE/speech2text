import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import torch
import torch.multiprocessing as mp
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)

from .align import align_transcription_and_diarization
from .audio import load_audio
from .diarize import diarize
from .settings import (available_whisper_models,
                       supported_languages_pretty,
                       DEFAULT_WHISPER_MODEL)
from .submit import parse_output_dir
from .transcribe import transcribe
from .utils import (convert_language_to_abbreviated_form)
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


def parse_output_file_stem(output_dir: str, input_file: str) -> Path:
    """
    Create output file stem from output directory and input file name.
    """
    return Path(output_dir) / Path(Path(input_file).name)


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
    combination, combination_done_time = align_transcription_and_diarization(
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
