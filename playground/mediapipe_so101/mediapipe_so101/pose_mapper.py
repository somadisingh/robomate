from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .types import HandFeatures, HandSample, Landmark, RobotTargets


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17


@dataclass(frozen=True)
class MappingConfig:
    wrist_flex_gain: float = 30.0
    wrist_roll_gain: float = 60.0
    gripper_open: float = 80.0
    gripper_closed: float = 20.0
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.40
    min_hand_width: float = 0.03

    def __post_init__(self) -> None:
        for field_name in (
            "wrist_flex_gain",
            "wrist_roll_gain",
            "gripper_open",
            "gripper_closed",
            "pinch_closed_ratio",
            "pinch_open_ratio",
            "min_hand_width",
        ):
            _validate_finite_number(field_name, getattr(self, field_name))

        if self.min_hand_width <= 0.0:
            raise ValueError("min_hand_width must be greater than 0")
        if self.pinch_open_ratio <= self.pinch_closed_ratio:
            raise ValueError("pinch_open_ratio must be greater than pinch_closed_ratio")


class PoseMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: HandFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: HandSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: HandSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral pose has not been captured")

        features = extract_features(sample, self.config)
        flex_delta = features.flex - self._neutral_features.flex
        roll_delta = _normalize_angle(features.roll - self._neutral_features.roll)
        gripper = self.config.gripper_closed + features.pinch_open * (
            self.config.gripper_open - self.config.gripper_closed
        )

        return RobotTargets(
            wrist_flex=self._neutral_targets.wrist_flex + flex_delta * self.config.wrist_flex_gain,
            wrist_roll=self._neutral_targets.wrist_roll + roll_delta * self.config.wrist_roll_gain,
            gripper=gripper,
        )


def extract_features(sample: HandSample, config: MappingConfig | None = None) -> HandFeatures:
    cfg = config or MappingConfig()
    landmarks = sample.landmarks
    if len(landmarks) < 21:
        raise ValueError("HandSample must contain 21 landmarks")

    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]

    for landmark in (wrist, thumb_tip, index_mcp, index_tip, middle_mcp, pinky_mcp):
        _validate_landmark(landmark)

    hand_width = _distance(index_mcp, pinky_mcp)
    if hand_width < cfg.min_hand_width:
        raise ValueError("Hand width is too small")

    roll = math.atan2(index_mcp.y - pinky_mcp.y, pinky_mcp.x - index_mcp.x)
    flex = (middle_mcp.y - wrist.y) / hand_width
    pinch_ratio = _distance(thumb_tip, index_tip) / hand_width
    pinch_open = _clamp(
        (pinch_ratio - cfg.pinch_closed_ratio) / (cfg.pinch_open_ratio - cfg.pinch_closed_ratio),
        0.0,
        1.0,
    )

    return HandFeatures(roll=roll, flex=flex, pinch_open=pinch_open)


def _distance(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _validate_landmark(landmark: Landmark) -> None:
    if not all(math.isfinite(value) for value in (landmark.x, landmark.y, landmark.z)):
        raise ValueError("Landmark coordinates must be finite")


def _validate_robot_targets(robot_targets: RobotTargets) -> None:
    if not all(
        math.isfinite(value)
        for value in (robot_targets.wrist_flex, robot_targets.wrist_roll, robot_targets.gripper)
    ):
        raise ValueError("Robot target values must be finite")


def _validate_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
