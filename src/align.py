from collections import defaultdict
import time

import whisperx

from .diarize import assign_word_speakers
from .settings import (wav2vec_models,
                       DEFAULT_COMPUTE_DEVICE) 
from .utils import seconds_to_human_readable_format

def align_transcription_and_diarization(
    transcription_segments, diarization_segments, file, language
) -> dict:
    """
    Combine transcription and diarization results:

    1. Convert transcription segments using wav2vec alignment so that each segment corresponds to a word
    2. For each transcribed word, find the most overlapping (in time) diarization/speaker segment

    If no diarization segment overlaps with a word, the speaker for that word is "SPEAKER_UNKNOWN".

    Parameters
    ----------
    transcription_segments : list
        Output of Whisper transcribe()
    diarization_segments : list
        Output of Pyannote diarization()
    file : str
        The input audio file
    language: str
        language in short format (e.g. "fi")

    Returns
    -------
    dict:
        The output object is a dict of lists:
        {
        "start" : [0.0, 4.5, 7.0],
        "end"   : [3.3, 6.0, 10.0],
        "transcription" : ["This is first first speaker segment", "This is the second", "This is from an unknown speaker"],
        "speaker": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_UNKNOWN"]
        }
    """

    # Convert transcription segments so that each segment corresponds to a word
    wav2vec_model_name = (
        wav2vec_models[language]
        if language in wav2vec_models
        else None
    )

    align_model, align_metadata = whisperx.load_align_model(
        language, DEFAULT_COMPUTE_DEVICE, model_name=wav2vec_model_name
    )

    transcription_segments = whisperx.align(
        transcription_segments,
        align_model,
        align_metadata,
        file,
        DEFAULT_COMPUTE_DEVICE,
    )

    # Assign speaker to transcribed word segments
    segments = assign_word_speakers(
        diarization_segments, transcription_segments["segments"]
    )

    # Reformat the result (return a dictionary of lists)
    result = defaultdict(list)
    for segment in segments:
        transcription_start = seconds_to_human_readable_format(segment["start"])
        transcription_end = seconds_to_human_readable_format(segment["end"])

        result["start"].append(transcription_start)
        result["end"].append(transcription_end)
        result["transcription"].append(segment["text"].strip())
        try:
            result["speaker"].append(segment["speaker"])
        except KeyError:
            result["speaker"].append("SPEAKER_UNKNOWN")

    return result, time.time()