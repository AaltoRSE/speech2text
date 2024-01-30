"""
Submit multiple audio files in a folder.
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path, PosixPath

import settings
from utils import add_durations, load_audio

# This is the speedup to realtime for transcribing the audio file.
# The real number is higher than 15, this is just to make sure the job has enough time to complete.
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
        default=os.getenv("SPEECH2TEXT_TMP"),
        help="Temporary folder. If not given, should be set as an environment variable.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_MEM",
        type=str,
        default=os.getenv("SPEECH2TEXT_MEM"),
        help="Requested memory per job. If not given, should be set as an environment variable.",
    )
    parser.add_argument(
        "--SPEECH2TEXT_CPUS_PER_TASK",
        type=str,
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
        help="Language. Optional.",
    )

    return parser


def get_existing_result_files(input_file, output_dir):
    existing_result_files, missing_result_files = [], []
    for suffix in [".csv", ".txt"]:
        output_file = Path(output_dir) / Path(Path(input_file).name).with_suffix(suffix)
        if output_file.is_file():
            existing_result_files.append(output_file)
        else:
            missing_result_files.append(output_file)

    return existing_result_files, missing_result_files


def parse_job_name(input_path):
    return Path(input_path).name


def parse_output_dir(input_path, create_if_not_exists=True):
    if Path(input_path).is_dir():
        output_dir = Path(input_path).absolute() / "results"
    elif Path(input_path).is_file():
        output_dir = Path(input_path).absolute().parent / "results"
    else:
        raise ValueError(f"Input path is not a file nor a directory: {input_path}")

    if create_if_not_exists:
        output_dir.mkdir(exist_ok=True, parents=True)

    return output_dir


def create_array_input_file(input_dir, output_dir, job_name, tmp_dir):
    print(f"Scan input audio files from: {input_dir}\n")
    input_files = []
    for input_file in Path(input_dir).glob("*.*"):
        existing, missing = get_existing_result_files(input_file, output_dir)
        if existing and not missing:
            print(
                f".. {input_file}: Skip since result files {[str(f) for f in existing]} exist"
            )
            continue
        print(f".. {input_file}: Submit")
        input_files.append(str(input_file))
    print()

    if not input_files:
        return

    tmp_file_array = (Path(tmp_dir) / str(job_name)).with_suffix(".json")
    Path(tmp_file_array).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_file_array, "w") as fout:
        json.dump(input_files, fout)

    return tmp_file_array


def estimate_job_time(input_path: PosixPath) -> str:
    """
    Estimate total run time based on input file/folder

    Parameters
    ----------
    input_path: PosixPath
        Input audio file or folder containing audio files.

    Returns
    -------
    Duration: str
        Total estimate time in HH:MM:SS format.
    """
    # Loading time for whisper + diarization pipeline
    PIPELINE_LOADING_TIME = "00:05:00"
    # Loading a 60 minute audio file takes ~5 seconds. This is an upper limit (equivalent to
    # loading a 24h file) to ensure sufficient time.
    AUDIO_LOADING_TIME = "00:01:00"

    total_duration = "00:00:00"
    total_loading = "00:00:00"

    input_files = []
    if Path(input_path).suffix == ".json":
        with open(input_path, "r") as fin:
            input_files = json.load(fin)
    else:
        input_files.append(str(input_path))

    for audio_file in input_files:
        _, duration = load_audio(audio_file)

        hours, minutes, seconds = map(int, duration.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        result_seconds = total_seconds / REALTIME_SPEEDUP

        if result_seconds < 60:
            result_seconds = 60

        result_hours, remainder = divmod(result_seconds, 3600)
        result_minutes, result_seconds = divmod(remainder, 60)

        duration = "{:02}:{:02}:{:02}".format(
            int(result_hours), int(result_minutes), int(result_seconds)
        )

        total_duration = add_durations(total_duration, duration)
        total_loading = add_durations(total_loading, AUDIO_LOADING_TIME)

    audio_processing_time = add_durations(total_duration, total_loading)
    return add_durations(PIPELINE_LOADING_TIME, audio_processing_time)


def create_sbatch_script_for_array_job(
    input_file, job_name, mem, cpus_per_task, time, email, tmp_dir
):
    with open(input_file, "r") as fin:
        array_length = len(json.load(fin))

    python_source_dir = Path(__file__).absolute().parent

    script = f"""#!/bin/bash
#SBATCH --array=0-{array_length-1}
#SBATCH --output="{tmp_dir}/{job_name}_%A_%a.out"
#SBATCH --job-name={job_name}
#SBATCH --mem={mem} 
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:1
#SBATCH --time={time}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
export OMP_NUM_THREADS={cpus_per_task}
export KMP_AFFINITY=granularity=fine,compact
python3 {python_source_dir}/speech2text.py {input_file}
"""
    tmp_file_sh = (Path(tmp_dir) / str(job_name)).with_suffix(".sh")
    Path(tmp_file_sh).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_file_sh, "w") as fout:
        fout.write(script)

    return tmp_file_sh


def submit_dir(args, job_name):
    # Prepare submission scripts
    output_dir = parse_output_dir(args.INPUT)
    tmp_file_array = create_array_input_file(
        args.INPUT, output_dir, job_name, args.SPEECH2TEXT_TMP
    )
    if tmp_file_array is None:
        print(
            f"Submission not necessary since no files in {args.INPUT} need processing\n"
        )
        return

    estimated_time = estimate_job_time(tmp_file_array)
    tmp_file_sh = create_sbatch_script_for_array_job(
        tmp_file_array,
        job_name,
        args.SPEECH2TEXT_MEM,
        args.SPEECH2TEXT_CPUS_PER_TASK,
        estimated_time,
        args.SPEECH2TEXT_EMAIL,
        args.SPEECH2TEXT_TMP,
    )

    # Submit
    cmd = f"sbatch {tmp_file_sh.absolute()}"
    cmd = shlex.split(cmd)
    subprocess.run(cmd)


def create_sbatch_script_for_single_file(
    input_file, job_name, mem, cpus_per_task, time, email, tmp_dir
):
    python_source_dir = Path(__file__).absolute().parent

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output="{tmp_dir}/{job_name}_%j.out"
#SBATCH --mem={mem} 
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:1
#SBATCH --time={time}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
python3 {python_source_dir}/speech2text.py {input_file}
"""

    tmp_file_sh = (Path(tmp_dir) / str(job_name)).with_suffix(".sh")
    Path(tmp_file_sh).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_file_sh, "w") as fout:
        fout.write(script)

    return tmp_file_sh


def submit_file(args, job_name):
    # Prepare submission scripts
    output_dir = parse_output_dir(args.INPUT)

    # Check if expected result files exist already
    existing_result_files, missing_result_files = get_existing_result_files(
        args.INPUT, output_dir
    )
    if existing_result_files and not missing_result_files:
        print(
            f"Submission not necessary as expected result files already exist:\n{' '.join([str(f) for f in existing_result_files])}"
        )
        return
    estimated_time = estimate_job_time(args.INPUT)
    tmp_file_sh = create_sbatch_script_for_single_file(
        args.INPUT,
        job_name,
        args.SPEECH2TEXT_MEM,
        args.SPEECH2TEXT_CPUS_PER_TASK,
        estimated_time,
        args.SPEECH2TEXT_EMAIL,
        args.SPEECH2TEXT_TMP,
    )

    # Submit
    cmd = f"sbatch {tmp_file_sh.absolute()}"
    cmd = shlex.split(cmd)
    subprocess.run(cmd)


def check_language(language):
    supported_languages = list(settings.supported_languages.keys())

    if language is None:
        print(
            f"""No language given. The language will be detected automatically. To specify language explicitly (recommended), use
              
    export SPEECH2TEXT_LANGUAGE=mylanguage

where mylanguage is one of:\n\n{' '.join(supported_languages)}\n"""
        )

        return True

    if language.lower() in supported_languages:
        print(f"Given language '{language}' is supported.\n")
        return True

    print(
        f"Submission failed: Given language '{language}' not found in supported languages:\n\n{' '.join(supported_languages)}\n"
    )

    return False


def check_email(email):
    if email is not None:
        print(f"Email notifications will be sent to: {email}\n")
    else:
        print(
            f"""Notifications will not be sent as no email address was specified. To specify email address, use
              
    export SPEECH2TEXT_EMAIL=my.name@aalto.fi\n"""
        )


def main():
    # Parse arguments
    parser = get_argument_parser()
    args = parser.parse_args()
    print(f"\nSubmit speech2text jobs with arguments:")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    print()

    # Check language
    if not check_language(args.SPEECH2TEXT_LANGUAGE):
        return

    # Check email
    check_email(args.SPEECH2TEXT_EMAIL)

    # Notify about temporary folder location
    print(
        f"Log files (.out) and batch submit scripts (.sh) will be written to: {args.SPEECH2TEXT_TMP}\n"
    )

    # Submit file or directory
    args.INPUT = Path(args.INPUT).absolute()
    job_name = parse_job_name(args.INPUT)
    if Path(args.INPUT).is_file():
        print(f"Input file: {args.INPUT}\n")
        submit_file(args, job_name)
    elif Path(args.INPUT).is_dir():
        print(f"Input directory: {args.INPUT}\n")
        submit_dir(args, job_name)
    else:
        print(
            ".. Submission failed: First argument needs to be an existing audio file or a directory with audio files.\n"
        )


if __name__ == "__main__":
    main()
