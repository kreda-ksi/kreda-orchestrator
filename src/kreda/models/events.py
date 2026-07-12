from dataclasses import dataclass
from typing import Optional


@dataclass
class AlignedSegment:
    timestamp_ms: int
    event_type: str
    filename: str
    transcript_chunk: str
    track_id: int
    spatial_hint: Optional[str] = None
