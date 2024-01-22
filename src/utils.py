import logging
import numpy as np
import pandas as pd
import torch
import subprocess
from pathlib import Path
from pyannote.audio import Pipeline
from typing import Optional, Union
from whisperx import load_audio

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


def get_audio_length(file: str) -> int:
    """
    Returns a length of an audio file in seconds.

    Parameters
    ----------
    file: str
        The audio file

    Returns
    -------
    length: int
        Audio duration in length
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "compact=print_section=0:nokey=1:escape=csv",
            "-show_entries",
            "format=duration",
            file,
        ]   
        duration = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio: {e.stderr.decode()} duration") from e
    
    return int(float(duration.decode()))


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
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': 16000 
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df