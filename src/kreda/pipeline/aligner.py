import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, cast
from faster_whisper import WhisperModel
from kreda.models.config import WhisperConfig


@dataclass
class AlignedSegment:
    timestamp_ms: int
    event_type: str
    filename: str
    transcript_chunk: str


def get_keyframes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        names=["t_ms", "track", "type", "changed_or_what", "detail"],
    )

    # drop all strandard frames
    events = df[
        (df["type"] != "FRAME") & (df["changed_or_what"].str.startswith("SAVE_"))
    ].copy()

    subtype = cast(pd.Series, events["changed_or_what"]).str.split("_").str[1]

    # filename = std::format("{}/track_{}_{}_{}.png", cfg.run_dir, track_id, stream_ms, reason);
    events["filename"] = (
        "track_"
        + events["track"].astype(str)
        + "_"
        + events["t_ms"].astype(str)
        + "_"
        + subtype
        + ".png"
    )

    return cast(pd.DataFrame, events)


def transcribe_audio(audio_path: Path, cfg: WhisperConfig) -> List[Dict]:
    print(f"Loading Whisper {cfg.model_size} ({cfg.device}, {cfg.compute_type})")

    model = WhisperModel(
        cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
    )

    print(f"Transcribing {audio_path}...")
    segments, _ = model.transcribe(
        str(audio_path), language=cfg.language, beam_size=cfg.beam_size
    )

    transcript_data = []
    for segment in segments:
        transcript_data.append(
            {
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "text": segment.text.strip(),
            }
        )

    return transcript_data


def sync_timestamps(
    keyframes: pd.DataFrame, audio_segments: List[Dict], cfg: WhisperConfig
) -> List[AlignedSegment]:
    aligned_data = []
    last_time = 0

    for row in keyframes.itertuples():
        curr_time = int(row.t_ms)
        curr_cutoff_ms = curr_time + cfg.lag_padding_ms

        chunk_text = []
        for seg in audio_segments:
            if seg["start_ms"] >= last_time and seg["start_ms"] < curr_cutoff_ms:
                chunk_text.append(seg["text"])

        aligned_data.append(
            AlignedSegment(
                timestamp_ms=curr_time,
                event_type=cast(str, row.changed_or_what),
                filename=cast(str, row.filename),
                transcript_chunk=" ".join(chunk_text),
            )
        )

        last_time = curr_cutoff_ms

    return aligned_data


def run(run_path: Path, csv_path: Path, cfg: WhisperConfig) -> List[AlignedSegment]:
    keyframes = get_keyframes(run_path / csv_path)
    audio_segments = transcribe_audio(run_path / "audio.wav", cfg)
    aligned_segments = sync_timestamps(keyframes, audio_segments, cfg)

    return aligned_segments
