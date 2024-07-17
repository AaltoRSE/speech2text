import pytest

from utils import seconds_to_human_readable_format


def test_seconds_to_human_readable_format_normal():
    # Test normal cases
    assert seconds_to_human_readable_format(6) == "00:00:06"
    assert seconds_to_human_readable_format(60) == "00:01:00"
    assert seconds_to_human_readable_format(3600) == "01:00:00"
    assert seconds_to_human_readable_format(3660) == "01:01:00"
    assert seconds_to_human_readable_format(3661) == "01:01:01"


def test_seconds_to_human_readable_format_edge_cases():
    # Test edge cases, like 0 seconds or a large number
    assert seconds_to_human_readable_format(0) == "00:00:00"
    assert (
        seconds_to_human_readable_format(86399) == "23:59:59"
    )  # One second before a full day


def test_seconds_to_human_readable_format_error_cases():
    # Test potential error cases, like negative seconds
    with pytest.raises(ValueError):
        seconds_to_human_readable_format(-1)
