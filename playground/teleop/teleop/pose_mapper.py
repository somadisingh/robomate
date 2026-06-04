from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .types import ArmSample, HandSample, Landmark, PoseLandmark, RobotTargets, TeleopSample


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17


@dataclass(frozen=True)
class MappingConfig:
    shoulder_pan_gain: float = 30.0
    shoulder_lift_gain: float = 30.0
    elbow_flex_gain: float = 30.0
    wrist_flex_gain: float = 30.0
    wrist_roll_gain: float = 60.0
    gripper_open: float = 80.0
    gripper_closed: float = 20.0
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.40
    min_hand_width: float = 0.03
    min_arm_segment: float = 0.05
    mirror_hand: bool = False

    def __post_init__(self) -> None:
        numeric_fields = (
            "shoulder_pan_gain",
            "shoulder_lift_gain",
            "elbow_flex_gain",
            "wrist_flex_gain",
            "wrist_roll_gain",
            "gripper_open",
            "gripper_closed",
            "pinch_closed_ratio",
            "pinch_open_ratio",
            "min_hand_width",
            "min_arm_segment",
        )
        for field_name in numeric_fields:
            _validate_finite_number(field_name, getattr(self, field_name))

        if self.min_hand_width <= 0.0:
            raise ValueError("min_hand_width must be greater than 0")
        if self.min_arm_segment <= 0.0:
            raise ValueError("min_arm_segment must be greater than 0")
        if self.pinch_open_ratio <= self.pinch_closed_ratio:
            raise ValueError("pinch_open_ratio must be greater than pinch_closed_ratio")


@dataclass(frozen=True)
class HandFeatures:
    roll: float
    flex: float
    pinch_open: float


class WristMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: HandFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: HandSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_wrist_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: HandSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral wrist sample has not been captured")

        features = extract_wrist_features(sample, self.config)
        mirror = -1.0 if self.config.mirror_hand else 1.0
        flex_delta = features.flex - self._neutral_features.flex
        roll_delta = mirror * _normalize_angle(features.roll - self._neutral_features.roll)
        gripper = self.config.gripper_closed + features.pinch_open * (
            self.config.gripper_open - self.config.gripper_closed
        )

        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan,
            shoulder_lift=self._neutral_targets.shoulder_lift,
            elbow_flex=self._neutral_targets.elbow_flex,
            wrist_flex=self._neutral_targets.wrist_flex + flex_delta * self.config.wrist_flex_gain,
            wrist_roll=self._neutral_targets.wrist_roll + roll_delta * self.config.wrist_roll_gain,
            gripper=gripper,
        )


def extract_wrist_features(sample: HandSample, config: MappingConfig | None = None) -> HandFeatures:
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

    hand_width = _distance3(index_mcp, pinky_mcp)
    if hand_width < cfg.min_hand_width:
        raise ValueError("Hand width is too small")

    roll = math.atan2(index_mcp.y - pinky_mcp.y, pinky_mcp.x - index_mcp.x)
    flex = (middle_mcp.y - wrist.y) / hand_width
    pinch_ratio = _distance3(thumb_tip, index_tip) / hand_width
    pinch_open = _clamp(
        (pinch_ratio - cfg.pinch_closed_ratio) / (cfg.pinch_open_ratio - cfg.pinch_closed_ratio),
        0.0,
        1.0,
    )

    return HandFeatures(roll=roll, flex=flex, pinch_open=pinch_open)


def _distance3(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _validate_landmark(landmark: Landmark) -> None:
    if not all(math.isfinite(value) for value in (landmark.x, landmark.y, landmark.z)):
        raise ValueError("Landmark coordinates must be finite")


def _validate_robot_targets(robot_targets: RobotTargets) -> None:
    values = (
        robot_targets.shoulder_pan,
        robot_targets.shoulder_lift,
        robot_targets.elbow_flex,
        robot_targets.wrist_flex,
        robot_targets.wrist_roll,
        robot_targets.gripper,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Robot target values must be finite")


def _validate_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ArmFeatures:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float


def extract_arm_features(sample: ArmSample, config: MappingConfig | None = None) -> ArmFeatures:
    cfg = config or MappingConfig()
    for landmark in (sample.shoulder, sample.elbow, sample.wrist):
        _validate_pose_landmark(landmark)

    upper_arm = (
        sample.elbow.x - sample.shoulder.x,
        sample.elbow.y - sample.shoulder.y,
        sample.elbow.z - sample.shoulder.z,
    )
    forearm = (
        sample.wrist.x - sample.elbow.x,
        sample.wrist.y - sample.elbow.y,
        sample.wrist.z - sample.elbow.z,
    )

    upper_arm_length = math.sqrt(sum(component * component for component in upper_arm))
    forearm_length = math.sqrt(sum(component * component for component in forearm))
    if upper_arm_length < cfg.min_arm_segment:
        raise ValueError("Upper arm segment is too short")
    if forearm_length < cfg.min_arm_segment:
        raise ValueError("Forearm segment is too short")

    cos_elbow = sum(u * f for u, f in zip(upper_arm, forearm)) / (upper_arm_length * forearm_length)
    cos_elbow = _clamp(cos_elbow, -1.0, 1.0)
    elbow_flex = math.acos(cos_elbow)

    horizontal = math.sqrt(upper_arm[0] ** 2 + upper_arm[2] ** 2)
    shoulder_lift = math.atan2(horizontal, upper_arm[1])
    shoulder_pan = math.atan2(upper_arm[0], upper_arm[2])

    return ArmFeatures(
        shoulder_pan=shoulder_pan,
        shoulder_lift=shoulder_lift,
        elbow_flex=elbow_flex,
    )


def _validate_pose_landmark(landmark: PoseLandmark) -> None:
    if not all(
        math.isfinite(value) for value in (landmark.x, landmark.y, landmark.z, landmark.visibility)
    ):
        raise ValueError("Pose landmark coordinates and visibility must be finite")


class ArmMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: ArmFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: ArmSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_arm_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: ArmSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral arm features have not been captured")

        features = extract_arm_features(sample, self.config)
        pan_delta = _normalize_angle(features.shoulder_pan - self._neutral_features.shoulder_pan)
        lift_delta = features.shoulder_lift - self._neutral_features.shoulder_lift
        elbow_delta = features.elbow_flex - self._neutral_features.elbow_flex

        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan
            + pan_delta * self.config.shoulder_pan_gain,
            shoulder_lift=self._neutral_targets.shoulder_lift
            + lift_delta * self.config.shoulder_lift_gain,
            elbow_flex=self._neutral_targets.elbow_flex
            + elbow_delta * self.config.elbow_flex_gain,
            wrist_flex=self._neutral_targets.wrist_flex,
            wrist_roll=self._neutral_targets.wrist_roll,
            gripper=self._neutral_targets.gripper,
        )


class TeleopMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.arm = ArmMapper(config)
        self.wrist = WristMapper(config)

    @property
    def neutral_ready(self) -> bool:
        return self.arm.neutral_ready and self.wrist.neutral_ready

    def capture_neutral(self, sample: TeleopSample, robot_targets: RobotTargets) -> None:
        if sample.arm is None or sample.hand is None:
            raise ValueError("TeleopMapper neutral capture requires both arm and hand samples")
        self.arm.capture_neutral(sample.arm, robot_targets)
        self.wrist.capture_neutral(sample.hand, robot_targets)

    def map(self, sample: TeleopSample) -> RobotTargets:
        if sample.arm is None or sample.hand is None:
            raise ValueError("TeleopMapper map requires both arm and hand samples")
        if not self.neutral_ready:
            raise RuntimeError("Neutral TeleopMapper sample has not been captured")

        arm_targets = self.arm.map(sample.arm)
        wrist_targets = self.wrist.map(sample.hand)
        return RobotTargets(
            shoulder_pan=arm_targets.shoulder_pan,
            shoulder_lift=arm_targets.shoulder_lift,
            elbow_flex=arm_targets.elbow_flex,
            wrist_flex=wrist_targets.wrist_flex,
            wrist_roll=wrist_targets.wrist_roll,
            gripper=wrist_targets.gripper,
        )
