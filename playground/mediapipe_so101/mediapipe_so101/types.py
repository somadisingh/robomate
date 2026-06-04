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

CONTROLLED_KEYS = ("wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
HELD_KEYS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos")


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandSample:
    landmarks: Sequence[Landmark]
    handedness: str
    confidence: float
    timestamp_ms: int


@dataclass(frozen=True)
class HandFeatures:
    roll: float
    flex: float
    pinch_open: float


@dataclass(frozen=True)
class RobotTargets:
    wrist_flex: float
    wrist_roll: float
    gripper: float

    def as_action(self) -> dict[str, float]:
        return {
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
