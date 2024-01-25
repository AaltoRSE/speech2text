import re
import math
import logging
import numpy as np
import pandas as pd
import torch
import subprocess
from pathlib import Path
from pyannote.audio import Pipeline
from typing import Optional, Union
from datetime import datetime, timedelta

SAMPLE_RATE = 16000

logger = logging.getLogger("__name__")

def seconds_to_human_readable_format(seconds):
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
    
    duration_pattern = re.compile(r'time=(\d{2}:\d{2}:\d{2})')
    durations = duration_pattern.findall(str(out.stderr))

    if durations:
        return audio, durations[-1]
    else:
        raise RuntimeError(f"Failed to get audio duration from {file}")


class DiarizationPipeline:
    def __init__(
        self,
        config_file,
        model_name="pyannote/speaker-diarization-3.1",
        auth_token=None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        if Path(config_file).is_file():
            logger.info(".. .. Local config file found")
            self.model = Pipeline.from_pretrained(config_file).to(device)
        elif auth_token:
            logger.info(".. .. Downloading config from HuggingFace")
            self.model = Pipeline.from_pretrained(model_name, use_auth_token=auth_token).to(device)
        else:
            logger.error(
            "One of these is required: local pyannote config file or environment variable AUTH_TOKEN to download model from HuggingFace hub"
        )
            raise ValueError

    def __call__(self, audio: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio, _ = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE 
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        return diarize_df
    

def add_durations(time1, time2):
    dt_format = "%H:%M:%S"
    time1_obj = datetime.strptime(time1, dt_format)
    time2_obj = datetime.strptime(time2, dt_format)

    result_time = time1_obj + timedelta(hours=time2_obj.hour, minutes=time2_obj.minute, seconds=time2_obj.second)

    result_time_str = result_time.strftime(dt_format)

    return result_time_str


def calculate_max_batch_size() -> int:
    """
    This function is experimental to maximize the batch size based on cuda available memory.
    Based on multiple experiments, batch size of 4 requires 1GB of VRAM with float16. WhisperX and Pyannote
    models each require 3GB of VRAM. Float32 requires double memory for each batch.

    Parameters
    ----------

    Returns
    -------
    batch_size:
        Maximum batch_size for fitting in CUDA
    """
    
    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    batch_size = 4 * math.pow(2, math.floor(math.log2(total_gpu_vram)))
    if torch.cuda.FloatTensor().dtype == torch.float32:
        batch_size /=  4
    
    return int(batch_size)