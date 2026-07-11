from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    lag_padding_ms: int


@dataclass(frozen=True)
class AssemblerConfig:
    diff_threshold_pixels: int
    max_occlusion_ratio: float
