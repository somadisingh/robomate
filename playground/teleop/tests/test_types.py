import math

import pytest

from teleop.types import (
    ACTION_KEYS,
    CONTROLLED_KEYS,
    ArmSample,
    FilterResult,
    FreezeReason,
    HandSample,
    Landmark,
    PoseLandmark,
    RobotTargets,
    TeleopSample,
)


def test_action_keys_cover_all_six_joints() -> None:
    assert ACTION_KEYS == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )


def test_controlled_keys_match_action_keys() -> None:
    assert CONTROLLED_KEYS == ACTION_KEYS


def test_robot_targets_as_action_returns_all_six_keys_in_order() -> None:
    targets = RobotTargets(
        shoulder_pan=1.0,
        shoulder_lift=2.0,
        elbow_flex=3.0,
        wrist_flex=4.0,
        wrist_roll=5.0,
        gripper=60.0,
    )

    action = targets.as_action()

    assert tuple(action.keys()) == ACTION_KEYS
    assert action["shoulder_pan.pos"] == 1.0
    assert action["shoulder_lift.pos"] == 2.0
    assert action["elbow_flex.pos"] == 3.0
    assert action["wrist_flex.pos"] == 4.0
    assert action["wrist_roll.pos"] == 5.0
    assert action["gripper.pos"] == 60.0


def test_freeze_reason_values_include_active_paused_neutral_tracking_stale() -> None:
    assert {reason.value for reason in FreezeReason} >= {
        "active",
        "paused",
        "neutral_missing",
        "tracking_lost",
        "stale_result",
    }


def test_filter_result_carries_targets_and_reason() -> None:
    targets = RobotTargets(0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    result = FilterResult(
        targets=targets,
        frozen=True,
        clamped_keys=("wrist_flex.pos",),
        reason=FreezeReason.PAUSED,
    )

    assert result.targets is targets
    assert result.frozen is True
    assert result.clamped_keys == ("wrist_flex.pos",)
    assert result.reason is FreezeReason.PAUSED


def test_landmark_defaults_z_to_zero() -> None:
    point = Landmark(x=0.1, y=0.2)
    assert point.z == 0.0


def test_pose_landmark_carries_visibility() -> None:
    point = PoseLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9)
    assert point.visibility == 0.9


def test_hand_sample_holds_21_landmarks_and_metadata() -> None:
    points = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    sample = HandSample(landmarks=points, handedness="Right", confidence=0.8, timestamp_ms=10)
    assert len(sample.landmarks) == 21
    assert sample.handedness == "Right"
    assert sample.confidence == 0.8
    assert sample.timestamp_ms == 10


def test_arm_sample_holds_shoulder_elbow_wrist_pose_landmarks_and_timestamp() -> None:
    shoulder = PoseLandmark(0.0, 0.0, 0.0, visibility=0.9)
    elbow = PoseLandmark(0.0, 0.3, 0.0, visibility=0.9)
    wrist = PoseLandmark(0.0, 0.6, 0.0, visibility=0.9)
    sample = ArmSample(
        shoulder=shoulder,
        shoulder_image_xy=(0.2, 0.2),
        elbow=elbow,
        elbow_image_xy=(0.4, 0.4),
        wrist=wrist,
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=20,
    )
    assert sample.shoulder is shoulder
    assert sample.elbow is elbow
    assert sample.wrist is wrist
    assert sample.shoulder_image_xy == (0.2, 0.2)
    assert sample.elbow_image_xy == (0.4, 0.4)
    assert sample.wrist_image_xy == (0.5, 0.5)
    assert sample.timestamp_ms == 20


def test_teleop_sample_holds_arm_and_hand() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=20,
    )
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.8,
        timestamp_ms=20,
    )
    sample = TeleopSample(arm=arm, hand=hand, timestamp_ms=20)
    assert sample.arm is arm
    assert sample.hand is hand
    assert sample.timestamp_ms == 20


def test_teleop_sample_allows_missing_hand_or_arm() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=20,
    )
    sample = TeleopSample(arm=arm, hand=None, timestamp_ms=20)
    assert sample.arm is arm
    assert sample.hand is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shoulder_pan": math.nan},
        {"gripper": math.inf},
    ],
)
def test_robot_targets_accepts_non_finite_values(kwargs: dict) -> None:
    base = dict(
        shoulder_pan=0.0,
        shoulder_lift=0.0,
        elbow_flex=0.0,
        wrist_flex=0.0,
        wrist_roll=0.0,
        gripper=50.0,
    )
    base.update(kwargs)
    targets = RobotTargets(**base)
    assert isinstance(targets, RobotTargets)
