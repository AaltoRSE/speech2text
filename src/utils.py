import numpy as np


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
