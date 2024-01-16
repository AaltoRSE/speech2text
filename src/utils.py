import numpy as np
import subprocess

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