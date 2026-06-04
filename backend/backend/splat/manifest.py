from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SplatArtifact:
    path: str
    size_bytes: int
    num_gaussians: int


@dataclass(frozen=True)
class CameraPathArtifact:
    path: str
    frame_count: int
    fps: float


@dataclass(frozen=True)
class SeedPointsArtifact:
    path: str
    point_count: int


@dataclass(frozen=True)
class TrainInfo:
    iterations: int
    gpu: str
    duration_seconds: float
    method: str = "splatfacto"


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class SplatManifest:
    splat: SplatArtifact
    camera_path: CameraPathArtifact
    seed_points: SeedPointsArtifact
    train: TrainInfo
    intrinsics: Intrinsics
    version: int = MANIFEST_VERSION
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Inline extras at the top level if provided; keep version as int.
        extras = payload.pop("extras") or {}
        payload.update(extras)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True) + "\n"

    def write(self, output_path: Path | str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    def db_summary(self) -> dict[str, Any]:
        """Compact subset suitable for the recording_analysis_jobs.summary jsonb."""
        return {
            "num_gaussians": self.splat.num_gaussians,
            "frame_count": self.camera_path.frame_count,
            "fps": self.camera_path.fps,
            "train_duration_seconds": self.train.duration_seconds,
            "iterations": self.train.iterations,
            "gpu": self.train.gpu,
        }
