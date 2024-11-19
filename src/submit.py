"""
Submit multiple audio files in a folder.
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import os
import re
import shlex
import subprocess
from argparse import Namespace
from pathlib import Path, PosixPath

import settings
from utils import (add_durations, convert_language_to_abbreviated_form,
                   load_audio, get_tmp_folder)

# This is the speedup to realtime for transcribing the audio file.
# The real number is higher than 15 (close to 25), this is just to make sure the job has enough time to complete.
REALTIME_SPEEDUP = 15


def get_argument_parser():
    parser = argparse.ArgumentParser(
        prog="Aalto speech2text submit script",
        description="",
    )
    parser.add_argument(
        "INPUT",
        type=str,
        help="Input audio file or folder containing audio files. Mandatory.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_TMP",
        type=str,
        default=get_tmp_folder(),
        help="Temporary folder. If not given, can be set as an environment variable. Optional, defaults to: /scratch/work/$USER/.speech2text/",
    )
    parser.add_argument(
        "--SPEECH2TEXT_MEM",
        type=str,
        default=None,
        help="Requested memory per job. If not given, should be set as an environment variable.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_CPUS_PER_TASK",
        type=int,
        default=os.getenv("SPEECH2TEXT_CPUS_PER_TASK"),
        help="Requested cpus per task. If not given, should be set as an environment variable.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_EMAIL",
        type=str,
        default=os.getenv("SPEECH2TEXT_EMAIL"),
        help="Send job notifications to this email. Optional.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_LANGUAGE",
        type=str,
        default=os.getenv("SPEECH2TEXT_LANGUAGE"),
        help="Language. Mandatory.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_WHISPER_MODEL",
        type=str,
        default=os.getenv("SPEECH2TEXT_WHISPER_MODEL"),
        help=f"Whisper model. Default is {settings.default_whisper_model}.",
    )

    return parser


def get_existing_result_files(input_file: str, output_dir: str) -> "tuple[list, list]":
    """
    For the input file or folder, check if the expected result files exist already in the output directory.

    Parameters
    ----------
    inpup_file: str
        Input audio file or folder containing audio files

    output_dir: str
        Output directory where the result files are located. Default is next the the input file/folder.

    Returns
    -------
    existing_result_files: list
        list of the existing result files
    missing_result_files: list
        list of the missing result files
    """
    existing_result_files, missing_result_files = [], []
    for suffix in [".csv", ".txt"]:
        output_file = Path(output_dir) / Path(Path(input_file).name).with_suffix(suffix)
        if output_file.is_file():
            existing_result_files.append(output_file)
        else:
            missing_result_files.append(output_file)

    return existing_result_files, missing_result_files


def parse_job_name(input_path: str) -> str:
    """
    Convert input file/folder to str and replace spaces with underscore.

    Parameters
    ----------
    input_path: str
        The input path for the audio files.

    Returns
    -------
    str
        The job name extracted from the input path.
    """
    return "speech2text_" + Path(input_path).name.replace(" ", "_")


def parse_output_dir(input_path: str, create_if_not_exists: bool = True) -> str:
    """
    Create the output directory for the results.

    Parameters
    ----------
    input_path: str
        The input path for the audio files.

    Returns
    -------
    output_dir: str
        The output path for the results.
    """
    if Path(input_path).is_dir():
        output_dir = Path(input_path).absolute() / "results"
    elif Path(input_path).is_file():
        output_dir = Path(input_path).absolute().parent / "results"
    else:
        raise ValueError(f"Input path is not a file nor a directory: {input_path}")

    if create_if_not_exists:
        output_dir.mkdir(exist_ok=True, parents=True)

    return output_dir


def create_array_input_file(
    input_list: list[PosixPath], output_dir: str) -> list[str]:
    """
    Process the input list and create a list of audio files which needs transcribing.

    Parameters
    ----------
    input_dir: list[PosixPath]
        A list of audio files.
    output_dir: str
        The output directory for the results.

    Returns
    -------
    input_files: list[str]
        The list of audio files needs transcribing.
    """
    input_files = []
    
    for input_file in input_list:
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(input_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning! Error processing {input_file}: {e}")
            continue
        if "Audio:" not in str(result.stderr):
            print(f".. {input_file}: Skip since it's not an audio file.")
            continue
        existing, missing = get_existing_result_files(input_file, output_dir)
        if existing and not missing:
            print(
                f".. {input_file}: Skip since result files {[str(f) for f in existing]} exist."
            )
            continue
        input_files.append(str(input_file))

    return input_files


def estimate_job_requirements(input_path: PosixPath) -> tuple[str, int]:
    """
    Estimate total run time based on input file.

    Parameters
    ----------
    input_path: PosixPath
        Input audio file .

    Returns
    -------
    Duration: str
        Total estimate time in HH:MM:SS format.
    str: int
        Maximum required memory in "X"G format.
    """
    # Loading time for whisperx + diarization + diarization pipeline
    PIPELINE_LOADING_TIME = "00:08:00"
    # Loading a 60 minute audio file takes ~5 seconds. This is an upper limit (equivalent to
    # loading a 24h file) to ensure sufficient time.
    AUDIO_LOADING_TIME = "00:01:00"

    #Whisper and Pyannote models require 3.5Gb of memory each
    PIPELINE_REQ_RAM = 7
    
    _, duration, file_size = load_audio(str(input_path))

    audio_processing_time = add_durations(duration, AUDIO_LOADING_TIME)

    # Whisper and Pyannote uses 12x of file size for the RAM
    # Transcription and Diarization tasks run in parallel, x2 memory is required
    req_ram =  PIPELINE_REQ_RAM + file_size * 12 * 2
    return add_durations(PIPELINE_LOADING_TIME, audio_processing_time), f"{req_ram}G"


def submit_job(args: Namespace, audio_files: list[PosixPath]):
    """
    Run sbatch command to submit the job to the cluster.

    Parameters
    ----------
    args: Namespace
        The arguments for the submit script.
    audio_files: list[PosixPath]
        A list of audio files to be processed.
    """
    # Prepare submission scripts
    output_dir = parse_output_dir(args.INPUT)

    audio_files = create_array_input_file(audio_files, output_dir)
    
    if not audio_files:
        print(
            f"Submission not necessary since no files in {args.INPUT} need processing\n"
        )
        return


    for f in audio_files:
        est_time, req_ram = estimate_job_requirements(f)
        
        # For debugging
        if args.SPEECH2TEXT_MEM:
            req_ram = args.SPEECH2TEXT_MEM
        
        tmp_file_sh = create_sbatch_script_for_single_file(
            f,
            parse_job_name(f),
            req_ram,
            args.SPEECH2TEXT_CPUS_PER_TASK,
            est_time,
            args.SPEECH2TEXT_EMAIL,
            args.SPEECH2TEXT_TMP,
        )

        # Log
        print(f".. {f}: Submit.")
        # Submit
        cmd = f"sbatch {tmp_file_sh.absolute()}"
        cmd = shlex.split(cmd)
        subprocess.run(cmd)
    
    # Log
    print(f"Results will be written to folder: '{output_dir}'\n")


def create_sbatch_script_for_single_file(
    input_file, job_name, mem, cpus_per_task, time, email, tmp_dir
):
    python_source_dir = Path(__file__).absolute().parent

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output="{tmp_dir}/{job_name}_%j.out"
#SBATCH --error="{tmp_dir}/{job_name}_%j.err"
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:1
#SBATCH --time={time}
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL
python3 {python_source_dir}/speech2text.py {input_file}

wait

python3 src/notification.py --to {email} --file_name {Path(input_file).name} --file_path {Path(input_file).parent / 'results'}
"""

    tmp_file_sh = (Path(tmp_dir) / str(job_name)).with_suffix(".sh")
    Path(tmp_file_sh).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_file_sh, "w") as fout:
        fout.write(script)

    return tmp_file_sh


def check_email(email: str):
    """
    Check if the given email is valid.

    Parameters
    ----------
    email: str
        The email to check.
    """
    pattern = r"^[A-Za-z]+\.+[A-Za-z]+@aalto.fi$"
    if email is not None:
        if re.match(pattern, email):
            print(f"Email notifications will be sent to: {email}\n")
        else:
            print("Invalid email address. Please provide an Aalto email address.\n")
    else:
        print(
            f"""Notifications will not be sent as no email address was specified. To specify email address, use
              
export SPEECH2TEXT_EMAIL=my.name@aalto.fi\n"""
        )


def check_whisper_model(name: str) -> bool:
    """
    Check if the given Whisper model is supported.

    Parameters
    ----------
    name: str
        The Whisper model to check.

    Returns
    -------
    Boolean:
        True if the Whisper model is supported, False otherwise.
    """
    if name is None:
        print(
            f"Whisper model not given, using default '{settings.default_whisper_model}'.\n"
        )
        return True

    elif name in settings.available_whisper_models:
        print(f"Given Whisper model '{name}' is available.\n")
        return True

    print(
        f"Submission failed: Given Whisper model '{name}' is not among available models:\n\n{' '.join(settings.available_whisper_models)}.\n"
    )

    return False


def main():
    # Parse arguments
    parser = get_argument_parser()
    args, unknown = parser.parse_known_args()

    # Join all parts of the INPUT argument to handle spaces
    if unknown:
        args.INPUT = ' '.join([args.INPUT] + unknown)
    else:
        args.INPUT = args.INPUT
    
    print(f"\nSubmit speech2text jobs with arguments:")
    for key, value in vars(args).items():
        print(f"\t{key}: '{value}'")
    print()

    # Check temporary folder
    Path(args.SPEECH2TEXT_TMP).mkdir(parents=True, exist_ok=True)
    print(f"Temporary folder: {args.SPEECH2TEXT_TMP}\n")

    # Check mandatory language argument
    language = convert_language_to_abbreviated_form(args.SPEECH2TEXT_LANGUAGE)
    if language:
        print(f"Language: {language}\n")
    else:
        print(
            f"""Language not given or not supported.

Please specify the language using

export SPEECH2TEXT_LANGUAGE=mylanguage

where 'mylanguage' is one of the supported languages:

{settings.supported_languages_pretty}
"""
        )
        return

    # Check email
    check_email(args.SPEECH2TEXT_EMAIL)

    # Check Whisper model name
    if not check_whisper_model(args.SPEECH2TEXT_WHISPER_MODEL):
        return

    # Notify about temporary folder location
    print(
        f"Log files (.out) and batch submit scripts (.sh) will be written to: '{args.SPEECH2TEXT_TMP}'\n"
    )

    # Submit file or directory
    args.INPUT = Path(args.INPUT).absolute()

    if args.INPUT.is_file():
        print(f"Input file: '{args.INPUT}'\n")
        files_to_submit = [args.INPUT]
    elif args.INPUT.is_dir():
        print(f"Input directory: '{args.INPUT}'\n")
        files_to_submit = list(args.INPUT.glob("*.*")) 
    else:
        print(
            f".. Submission failed: First argument needs to be an existing audio file or a directory with audio files.\n \
            The input was set to '{args.INPUT}'"
        )
        return
    
    submit_job(args, files_to_submit)


if __name__ == "__main__":
    main()
