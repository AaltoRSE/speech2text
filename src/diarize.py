import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import pandas as pd
from pyannote.audio import Pipeline

from .utils import load_audio, SAMPLE_RATE

logger = logging.getLogger("__name__")


class DiarizationPipeline:
    def __init__(
        self,
        config_file: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
        auth_token: str = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        if Path(config_file).is_file():
            logger.info(f".. .. Loading local config file: {config_file}")
            self.model = Pipeline.from_pretrained(config_file).to(device)
        elif auth_token:
            logger.info(".. .. Downloading config file from HuggingFace")
            self.model = Pipeline.from_pretrained(
                model_name, use_auth_token=auth_token
            ).to(device)
        else:
            logger.error(
                "One of these is required: local pyannote config file or environment variable AUTH_TOKEN to download model from HuggingFace hub"
            )
            raise ValueError

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        if isinstance(audio, str):
            audio, _ = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        segments = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        return diarize_df


def assign_word_speakers(diarize_df, transcript_segments):
    """
    Assign speakers to words and segments in a transcript based on diarization results.

    Args:
        diarize_df (pd.DataFrame): The diarization dataframe.
        transcript_segments (list): The list of transcript segments.

    Returns:
        list: The list of transcript segments with assigned speakers.
    """
    for seg in transcript_segments:
        # assign speaker to segments
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        dia_tmp = diarize_df[diarize_df["intersection"] > 0]

        if len(dia_tmp) > 0:
            # sum over speakers if there are many speakers
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

        # assign speaker to each words
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])
                    dia_tmp = diarize_df[diarize_df["intersection"] > 0]

                    if len(dia_tmp) > 0:
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker

    return transcript_segments

