import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("__name__")

def write_result_to_csv_file(result: dict, output_file_stem: Path):
    """
    Write the transcription + diarization result to a CSV file.

    Parameters
    ----------
    result : dict
        Dictionary of lists for start, end, speaker, and transcription.
    output_file_stem : Path
        Output file stem.
    """
    df = pd.DataFrame.from_dict(result)
    output_file = str(Path(output_file_stem).with_suffix(".csv"))
    df.to_csv(
        output_file,
        sep=",",
        index=False,
        encoding="utf-8",
    )
    logger.info(f".. .. Wrote CSV output to: {output_file}")


def write_result_to_txt_file(result: dict, output_file_stem: Path):
    """
    Write the transcription + diarization result to a text file.

    Parameters
    ----------
    result : dict
        Dictionary of lists for start, end, speaker, and transcription.
    output_file_stem : Path
        Output file stem.
    """
    # Group lines by speaker
    all_lines_grouped_by_speaker = []
    lines_speaker = []
    prev_speaker = None
    for start, end, speaker, transcription in zip(
        result["start"],
        result["end"],
        result["speaker"],
        result["transcription"],
    ):
        if speaker != prev_speaker and lines_speaker:
            all_lines_grouped_by_speaker.append(lines_speaker)
            lines_speaker = []
        lines_speaker.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "transcription": transcription,
            }
        )
        prev_speaker = speaker

    # Append remainders
    if lines_speaker:
        all_lines_grouped_by_speaker.append(lines_speaker)

    # Write out
    lines_out = []
    for lines_speaker in all_lines_grouped_by_speaker:
        start = lines_speaker[0]["start"]  # first start time in group
        end = lines_speaker[-1]["end"]  # last end time in group
        speaker = lines_speaker[0]["speaker"]  # same speaker across group
        meta_info_line = f"({start} - {end}) {speaker}\n\n"
        lines_out.append(meta_info_line)
        transcriptions = (
            " ".join([line["transcription"] for line in lines_speaker]) + "\n\n"
        )
        lines_out.append(transcriptions)

    output_file = str(Path(output_file_stem).with_suffix(".txt"))
    with open(output_file, "w") as fout:
        for line in lines_out:
            fout.write(line)

    logger.info(f".. .. Wrote TXT output to: {output_file}")
