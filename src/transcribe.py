import gc 
import logging
import math
import subprocess
import time

import torch
import whisperx
from whisperx.types import TranscriptionResult
from typing import Optional, Union

from .settings import (available_whisper_models,
                       DEFAULT_WHISPER_MODEL, 
                       DEFAULT_COMPUTE_DEVICE)

logger = logging.getLogger("__name__")


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
