import pandas as pd
from kreda.pipeline.aligner import sync_timestamps
from kreda.models.config import WhisperConfig


def test_sync_timestamps_syncing_boundaries():
    keyframes = pd.DataFrame(
        [
            {"t_ms": 10000, "changed_or_what": "SAVE_slide", "filename": "slide1.png"},
            {"t_ms": 20000, "changed_or_what": "SAVE_slide", "filename": "slide2.png"},
        ]
    )

    audio_segments = [
        {"start_ms": 8000, "text": "Audio before slide 1 save."},
        {"start_ms": 12000, "text": "Audio just after slide 1 save (wrap up)."},
        {"start_ms": 16000, "text": "Audio starting slide 2 material."},
        {"start_ms": 22000, "text": "Audio just after slide 2 save."},
    ]

    mock_cfg = WhisperConfig(
        model_size="",
        device="",
        compute_type="",
        language="",
        beam_size=0,
        lag_padding_ms=5000,
    )

    segments = sync_timestamps(keyframes, audio_segments, mock_cfg)

    assert len(segments) == 2

    assert segments[0].filename == "slide1.png"
    assert "Audio before slide 1 save." in segments[0].transcript_chunk
    assert "Audio just after slide 1 save (wrap up)." in segments[0].transcript_chunk
    assert "Audio starting slide 2 material." not in segments[0].transcript_chunk
    assert "Audio just after slide 2 save." not in segments[0].transcript_chunk

    assert segments[1].filename == "slide2.png"
    assert "Audio before slide 1 save." not in segments[1].transcript_chunk
    assert (
        "Audio just after slide 1 save (wrap up)." not in segments[1].transcript_chunk
    )
    assert "Audio starting slide 2 material." in segments[1].transcript_chunk
    assert "Audio just after slide 2 save." in segments[1].transcript_chunk
