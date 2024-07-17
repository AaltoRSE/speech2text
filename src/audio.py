import re
import subprocess
import sys
import math

import numpy as np

SAMPLE_RATE = 16000

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
