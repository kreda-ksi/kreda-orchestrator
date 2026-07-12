from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str
    device: str
    compute_type: str
    input_language: str
    beam_size: int
    lag_padding_ms: int


@dataclass(frozen=True)
class AssemblerConfig:
    diff_threshold_pixels: int
    max_occlusion_ratio: float


class SessionPreset(str, Enum):
    lecture = "lecture"
    exercises = "exercises"
    seminar = "seminar"


@dataclass(frozen=True)
class SynthesizerConfig:
    input_language: str
    output_language: str
    course_domain: str
    preset: SessionPreset


@dataclass(frozen=True)
class GeneratorConfig:
    model: str
    api_key: str | None
    base_url: str | None
    chunk_size: int
    context_length: int
