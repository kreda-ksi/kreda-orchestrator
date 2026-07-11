from dataclasses import dataclass


@dataclass
class AlignedSegment:
    timestamp_ms: int
    event_type: str
    filename: str
    transcript_chunk: str
