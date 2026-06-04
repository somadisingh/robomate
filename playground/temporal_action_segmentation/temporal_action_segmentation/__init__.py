"""VITRA-inspired temporal action segmentation playground."""

from temporal_action_segmentation.segmentation import (
    Segment,
    SegmentationResult,
    interpolate_missing,
    speed_minima_segments,
)

__all__ = [
    "Segment",
    "SegmentationResult",
    "interpolate_missing",
    "speed_minima_segments",
]
