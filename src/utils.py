import logging
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline

import settings

SAMPLE_RATE = 16000

logger = logging.getLogger("__name__")


def seconds_to_human_readable_format(seconds: int) -> str:
    """
    Convert seconds to human readable string.

    Examples:

    seconds_to_human_readable(6) = "00:00:06"
    seconds_to_human_readable(60) = "00:01:00"
    seconds_to_human_readable(3600) = "01:00:00"
    seconds_to_human_readable(3660) = "01:01:00"
    seconds_to_human_readable(3661) = "01:01:01"

    """
    if seconds < 0:
        raise ValueError("Seconds argument needs to be positive.")

    SECONDS_IN_HOUR = 3600
    SECONDS_IN_MINUTE = 60
    hours = int(np.floor(seconds / SECONDS_IN_HOUR))
    minutes = int(np.floor((seconds - hours * SECONDS_IN_HOUR) / SECONDS_IN_MINUTE))
    seconds = int(
        round(seconds - hours * SECONDS_IN_HOUR - minutes * SECONDS_IN_MINUTE)
    )

    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_tmp_folder():
    ood_folder = os.getcwd()
    user = os.getenv("USER")
    return ood_folder if os.getenv('SPEECH2TEXT_ONDEMAND') else f"/scratch/work/{user}/.speech2text/"


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    Audio: NumPy Array
        Containing the audio waveform, in float32 dtype.
    Duration: str
        Audio Duration in HH:MM:SS
    File_Size: int
        File size in Gb
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    audio = np.frombuffer(out.stdout, np.int16).flatten().astype(np.float32) / 32768.0

    duration_pattern = re.compile(r"time=(\d{2}:\d{2}:\d{2})")
    durations = duration_pattern.findall(str(out.stderr))

    #Estimate the file size in Gb
    file_size = sys.getsizeof(audio) / 1024 / 1024 / 1024 

    if durations:
        return audio, durations[-1], math.ceil(file_size)
    else:
        raise RuntimeError(f"Failed to get audio duration from {file}")


class DiarizationPipeline:
    def __init__(
        self,
        config_file: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
        auth_token: str = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        if Path(config_file).is_file():
            logger.info(f".. .. Loading local config file: {config_file}")
            self.model = Pipeline.from_pretrained(config_file).to(device)
        elif auth_token:
            logger.info(".. .. Downloading config file from HuggingFace")
            self.model = Pipeline.from_pretrained(
                model_name, use_auth_token=auth_token
            ).to(device)
        else:
            logger.error(
                "One of these is required: local pyannote config file or environment variable AUTH_TOKEN to download model from HuggingFace hub"
            )
            raise ValueError

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        if isinstance(audio, str):
            audio, _ = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        segments = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        return diarize_df


def assign_word_speakers(diarize_df, transcript_segments):
    """
    Assign speakers to words and segments in a transcript based on diarization results.

    Args:
        diarize_df (pd.DataFrame): The diarization dataframe.
        transcript_segments (list): The list of transcript segments.

    Returns:
        list: The list of transcript segments with assigned speakers.
    """
    for seg in transcript_segments:
        # assign speaker to segments
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        dia_tmp = diarize_df[diarize_df["intersection"] > 0]

        if len(dia_tmp) > 0:
            # sum over speakers if there are many speakers
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

        # assign speaker to each words
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])
                    dia_tmp = diarize_df[diarize_df["intersection"] > 0]

                    if len(dia_tmp) > 0:
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker

    return transcript_segments


def add_durations(time1: str, time2: str) -> str:
    """
    Adds two time durations and returns the result.

    Args:
        time1 (str): The first time duration in the format "HH:MM:SS".
        time2 (str): The second time duration in the format "HH:MM:SS".

    Returns:
        str: The sum of the two time durations in the format "HH:MM:SS".
    """
    dt_format = "%H:%M:%S"
    time1_obj = datetime.strptime(time1, dt_format)
    time2_obj = datetime.strptime(time2, dt_format)

    result_time = time1_obj + timedelta(
        hours=time2_obj.hour, minutes=time2_obj.minute, seconds=time2_obj.second
    )

    result_time_str = result_time.strftime(dt_format)

    return result_time_str


def calculate_max_batch_size() -> int:
    """
    This function is experimental to maximize the batch size based on cuda available memory.
    Based on multiple experiments, batch size of 4 requires 1GB of VRAM with float16. WhisperX and Pyannote
    models each require 3GB of VRAM. Float32 requires double memory for each batch.

    Parameters
    ----------
    None

    Returns
    -------
    batch_size:
        Maximum batch_size for fitting in a CUDA device.
    """
    total_gpu_vram = (
        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    )
    batch_size = 4 * math.pow(2, math.floor(math.log2(total_gpu_vram)))
    if torch.cuda.FloatTensor().dtype == torch.float32:
        batch_size /= 4

    return int(batch_size)


def convert_language_to_abbreviated_form(language: str) -> str:
    """
    Convert language to abbreviated form if it is given in long form.

    Parameters
    ----------
    language : str
        The language to be converted. It can be given in long form.

    Returns
    -------
    str
        The language in abbreviated form (lower-cased two-letter abbreviation) if it is given in long form.
        If the language is already in abbreviated form, it will be returned as its lower-cased form.
        If the conversion cannot be made, None will be returned.
    """
    if not language:
        return None

    # Language is given in OK long form: convert to short form (two-letter abbreviation)
    elif language.lower() in settings.supported_languages.keys():
        return settings.supported_languages[language.lower()]

    # Language is given in OK short form (two-letter abbreviation)
    elif language.lower() in settings.supported_languages.values():
        return language.lower()

    # Conversion cannot be made
    return None
