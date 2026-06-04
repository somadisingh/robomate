from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoSample:
    index: int
    source_frame: int
    time_sec: float


@dataclass(frozen=True)
class HandTrack:
    hand: str
    fps: float
    points: np.ndarray
    confidence: np.ndarray

    @property
    def visible(self) -> np.ndarray:
        return np.isfinite(self.points).all(axis=1) & np.isfinite(self.confidence)


@dataclass(frozen=True)
class VideoTracks:
    video_id: str
    video_path: Path
    width: int
    height: int
    source_fps: float
    sample_fps: float
    samples: list[VideoSample]
    tracks: dict[str, HandTrack]
