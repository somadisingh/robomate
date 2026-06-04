import pytest

import main
from teleop.pose_mapper import MappingConfig, TeleopMapper
from teleop.robot_backend import DryRunBackend
from teleop.safety import SafetyConfig, TargetFilter
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


def baseline() -> RobotTargets:
    return RobotTargets(0.0, 0.0, 0.0, 0.0, 0.0, 80.0)


def safety_config() -> SafetyConfig:
    return SafetyConfig(
        limits={
            "shoulder_pan.pos": (-20.0, 20.0),
            "shoulder_lift.pos": (-20.0, 20.0),
            "elbow_flex.pos": (-25.0, 25.0),
            "wrist_flex.pos": (-15.0, 15.0),
            "wrist_roll.pos": (-25.0, 25.0),
            "gripper.pos": (15.0, 85.0),
        },
        max_delta={key: 2.0 for key in CONTROLLED_KEYS},
        smoothing=0.35,
        stale_timeout_ms=200,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
    )


def mapping_config() -> MappingConfig:
    return MappingConfig(
        shoulder_pan_gain=1.0,
        shoulder_lift_gain=1.0,
        elbow_flex_gain=1.0,
        wrist_flex_gain=30.0,
        wrist_roll_gain=60.0,
        gripper_open=80.0,
        gripper_closed=20.0,
        pinch_closed_ratio=0.35,
        pinch_open_ratio=1.40,
    )


def sample(timestamp_ms: int = 1000) -> TeleopSample:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=timestamp_ms,
    )
    points = [Landmark(0.5, 0.65, 0.0) for _ in range(21)]
    points[5] = Landmark(0.45, 0.55, 0.0)
    points[17] = Landmark(0.55, 0.55, 0.0)
    points[9] = Landmark(0.50, 0.45, 0.0)
    points[4] = Landmark(0.42, 0.40, 0.0)
    points[8] = Landmark(0.58, 0.40, 0.0)
    hand = HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=timestamp_ms)
    return TeleopSample(arm=arm, hand=hand, timestamp_ms=timestamp_ms)


def test_sample_is_usable_returns_true_for_good_sample() -> None:
    assert main.sample_is_usable(
        sample(),
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_pose_visibility_below_threshold() -> None:
    sample_with_low_visibility = sample()
    bad_arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.2),
        shoulder_image_xy=sample_with_low_visibility.arm.shoulder_image_xy,
        elbow=sample_with_low_visibility.arm.elbow,
        elbow_image_xy=sample_with_low_visibility.arm.elbow_image_xy,
        wrist=sample_with_low_visibility.arm.wrist,
        wrist_image_xy=sample_with_low_visibility.arm.wrist_image_xy,
        timestamp_ms=sample_with_low_visibility.arm.timestamp_ms,
    )
    s = TeleopSample(arm=bad_arm, hand=sample_with_low_visibility.hand, timestamp_ms=1000)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_arm_missing() -> None:
    s = TeleopSample(arm=None, hand=sample().hand, timestamp_ms=1000)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_stale() -> None:
    assert not main.sample_is_usable(
        sample(timestamp_ms=500),
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_underlying_samples_are_stale() -> None:
    # Regression: TeleopSample.timestamp_ms must reflect the underlying MediaPipe
    # callback time, not the loop clock — otherwise stale-result detection is dead.
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=500,
    )
    points = [Landmark(0.5, 0.65, 0.0) for _ in range(21)]
    points[5] = Landmark(0.45, 0.55, 0.0)
    points[17] = Landmark(0.55, 0.55, 0.0)
    points[9] = Landmark(0.50, 0.45, 0.0)
    points[4] = Landmark(0.42, 0.40, 0.0)
    points[8] = Landmark(0.58, 0.40, 0.0)
    hand = HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=500)
    stale_sample = TeleopSample(arm=arm, hand=hand, timestamp_ms=500)

    assert not main.sample_is_usable(
        stale_sample,
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_handle_neutral_capture_rejects_unusable_sample() -> None:
    mapper = TeleopMapper(mapping_config())
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState()
    result = main.handle_neutral_capture(
        sample=TeleopSample(arm=None, hand=None, timestamp_ms=1000),
        now_ms=1000,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=baseline(),
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
        state=state,
    )
    assert result is target_filter
    assert mapper.neutral_ready is False
    assert state.notice is not None and "neutral rejected" in state.notice


def test_handle_neutral_capture_succeeds_with_usable_sample() -> None:
    mapper = TeleopMapper(mapping_config())
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState()
    result = main.handle_neutral_capture(
        sample=sample(),
        now_ms=1000,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=baseline(),
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
        state=state,
    )
    assert mapper.neutral_ready is True
    assert state.notice == "neutral captured"


def test_handle_sync_toggle_disabled_when_send_failed() -> None:
    state = main.LoopState(send_failed=True)
    main.handle_sync_toggle(state)
    assert state.sync_enabled is False
    assert "send failed" in (state.notice or "")


def test_handle_sync_toggle_flips_when_send_ok() -> None:
    state = main.LoopState()
    main.handle_sync_toggle(state)
    assert state.sync_enabled is True
    main.handle_sync_toggle(state)
    assert state.sync_enabled is False


def test_arm_image_landmarks_returns_all_three_pose_landmarks() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow_image_xy=(0.4, 0.4),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=1000,
    )
    sample = TeleopSample(arm=arm, hand=None, timestamp_ms=1000)

    landmarks = main._arm_image_landmarks("right", sample)

    assert landmarks is not None
    assert set(landmarks.keys()) == {"shoulder", "elbow", "wrist"}
    assert landmarks["shoulder"] == (0.2, 0.2)
    assert landmarks["elbow"] == (0.4, 0.4)
    assert landmarks["wrist"] == (0.5, 0.5)


def test_arm_image_landmarks_returns_none_when_arm_missing() -> None:
    sample = TeleopSample(arm=None, hand=None, timestamp_ms=1000)
    assert main._arm_image_landmarks("right", sample) is None


def test_handle_backend_send_locks_off_on_failure() -> None:
    class BrokenBackend(DryRunBackend):
        def send(self, _targets):
            raise RuntimeError("port closed")

    backend = BrokenBackend(default_gripper=80.0)
    backend.connect()
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState(sync_enabled=True)
    result = FilterResult(
        targets=baseline(),
        frozen=False,
        clamped_keys=(),
        reason=FreezeReason.ACTIVE,
    )
    new_result = main.handle_backend_send(backend, result, target_filter, state)
    assert state.sync_enabled is False
    assert state.send_failed is True
    assert new_result.frozen is True
    assert new_result.reason is FreezeReason.PAUSED
