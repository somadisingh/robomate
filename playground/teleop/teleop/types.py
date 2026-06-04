from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

CONTROLLED_KEYS = ACTION_KEYS


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class HandSample:
    landmarks: Sequence[Landmark]
    handedness: str
    confidence: float
    timestamp_ms: int


@dataclass(frozen=True)
class ArmSample:
    shoulder: PoseLandmark
    shoulder_image_xy: tuple[float, float]
    elbow: PoseLandmark
    elbow_image_xy: tuple[float, float]
    wrist: PoseLandmark
    wrist_image_xy: tuple[float, float]
    timestamp_ms: int


@dataclass(frozen=True)
class TeleopSample:
    arm: ArmSample | None
    hand: HandSample | None
    timestamp_ms: int


@dataclass(frozen=True)
class RobotTargets:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float

    def as_action(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": self.shoulder_pan,
            "shoulder_lift.pos": self.shoulder_lift,
            "elbow_flex.pos": self.elbow_flex,
            "wrist_flex.pos": self.wrist_flex,
            "wrist_roll.pos": self.wrist_roll,
            "gripper.pos": self.gripper,
        }


class FreezeReason(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    NEUTRAL_MISSING = "neutral_missing"
    TRACKING_LOST = "tracking_lost"
    STALE_RESULT = "stale_result"


@dataclass(frozen=True)
class FilterResult:
    targets: RobotTargets
    frozen: bool
    clamped_keys: tuple[str, ...]
    reason: FreezeReason
