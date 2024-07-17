import logging
import math
from datetime import datetime, timedelta

import numpy as np
import torch

from .settings import supported_languages

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
    elif language.lower() in supported_languages.keys():
        return supported_languages[language.lower()]

    # Language is given in OK short form (two-letter abbreviation)
    elif language.lower() in supported_languages.values():
        return language.lower()

    # Conversion cannot be made
    return None
