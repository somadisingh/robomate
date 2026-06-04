from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from main import (
    LoopState,
    handle_backend_send,
    handle_neutral_capture,
    handle_sync_toggle,
    make_safety_config,
    neutral_rejection_reason,
    sample_is_usable,
    validate_args,
)
from mediapipe_so101.pose_mapper import MappingConfig, PoseMapper
from mediapipe_so101.safety import SafetyConfig, TargetFilter
from mediapipe_so101.types import FilterResult, FreezeReason, HandSample, Landmark, RobotTargets


def make_sample(*, confidence: float = 0.9, timestamp_ms: int = 100) -> HandSample:
    landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    landmarks[0] = Landmark(0.50, 0.80, 0.0)
    landmarks[4] = Landmark(0.45, 0.45, 0.0)
    landmarks[5] = Landmark(0.40, 0.40, 0.0)
    landmarks[8] = Landmark(0.55, 0.45, 0.0)
    landmarks[9] = Landmark(0.50, 0.35, 0.0)
    landmarks[17] = Landmark(0.62, 0.40, 0.0)
    return HandSample(landmarks=landmarks, handedness="Right", confidence=confidence, timestamp_ms=timestamp_ms)


def make_filter() -> TargetFilter:
    return TargetFilter(
        SafetyConfig(
            limits={
                "wrist_flex.pos": (-25.0, 25.0),
                "wrist_roll.pos": (-45.0, 45.0),
                "gripper.pos": (15.0, 85.0),
            },
            max_delta={
                "wrist_flex.pos": 4.0,
                "wrist_roll.pos": 4.0,
                "gripper.pos": 4.0,
            },
            smoothing=0.35,
            stale_timeout_ms=150,
        ),
        RobotTargets(0.0, 0.0, 50.0),
    )


def valid_args(**overrides) -> argparse.Namespace:
    values = {
        "fps": 5,
        "width": 640,
        "height": 480,
        "max_hands": 1,
        "deadman_key": "",
        "deadman_grace_ms": 175,
        "detection_confidence": 0.5,
        "presence_confidence": 0.5,
        "tracking_confidence": 0.5,
        "min_hand_confidence": 0.45,
        "wrist_flex_gain": 30.0,
        "wrist_roll_gain": 60.0,
        "gripper_open": 80.0,
        "gripper_closed": 20.0,
        "pinch_closed_ratio": 0.35,
        "pinch_open_ratio": 1.40,
        "wrist_flex_limit": 25.0,
        "wrist_roll_limit": 45.0,
        "gripper_min": 15.0,
        "gripper_max": 85.0,
        "max_delta": 4.0,
        "smoothing": 0.35,
        "stale_timeout_ms": 150,
        "enable_robot": False,
        "robot_port": None,
        "robot_id": None,
        "calibration_dir": Path("unused"),
        "max_relative_target": 5.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_args_rejects_mismatched_usb_serial_and_robot_id(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "so101_5AE60843881.json").write_text("{}")
    args = valid_args(
        enable_robot=True,
        robot_port="/dev/cu.usbmodem5B140317341",
        robot_id="so101_5AE60843881",
        calibration_dir=calibration_dir,
    )

    with pytest.raises(SystemExit, match="serial '5B140317341'"):
        validate_args(args)


def test_validate_args_allows_matching_usb_serial_and_robot_id(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "so101_5B140317341.json").write_text("{}")
    args = valid_args(
        enable_robot=True,
        robot_port="/dev/tty.usbmodem5B140317341",
        robot_id="so101_5B140317341",
        calibration_dir=calibration_dir,
    )

    validate_args(args)


def test_safety_config_allows_startup_gripper_outside_requested_motion_bounds() -> None:
    args = valid_args(gripper_min=15.0, gripper_max=85.0)
    startup_targets = RobotTargets(wrist_flex=0.0, wrist_roll=0.0, gripper=1.02)

    target_filter = TargetFilter(make_safety_config(args, startup_targets), startup_targets)

    assert target_filter.last_targets == startup_targets
    assert target_filter.config.limits["gripper.pos"] == (1.02, 85.0)


def test_neutral_rejection_reason_rejects_low_confidence_sample() -> None:
    sample = make_sample(confidence=0.44, timestamp_ms=100)

    assert not sample_is_usable(sample, now_ms=110, min_hand_confidence=0.45, stale_timeout_ms=150)
    assert (
        neutral_rejection_reason(sample, now_ms=110, min_hand_confidence=0.45, stale_timeout_ms=150)
        is FreezeReason.TRACKING_LOST
    )


def test_neutral_capture_rejects_stale_sample_without_capturing() -> None:
    mapper = PoseMapper(MappingConfig())
    target_filter = make_filter()
    state = LoopState()

    returned_filter = handle_neutral_capture(
        sample=make_sample(timestamp_ms=100),
        now_ms=251,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=RobotTargets(0.0, 0.0, 50.0),
        min_hand_confidence=0.45,
        stale_timeout_ms=150,
        state=state,
    )

    assert returned_filter is target_filter
    assert not mapper.neutral_ready
    assert state.notice == "neutral rejected: stale_result"


def test_neutral_capture_rejects_low_confidence_sample_without_capturing() -> None:
    mapper = PoseMapper(MappingConfig())
    target_filter = make_filter()
    state = LoopState()

    returned_filter = handle_neutral_capture(
        sample=make_sample(confidence=0.44, timestamp_ms=100),
        now_ms=100,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=RobotTargets(0.0, 0.0, 50.0),
        min_hand_confidence=0.45,
        stale_timeout_ms=150,
        state=state,
    )

    assert returned_filter is target_filter
    assert not mapper.neutral_ready
    assert state.notice == "neutral rejected: tracking_lost"


def test_neutral_capture_rejects_missing_sample_without_capturing() -> None:
    mapper = PoseMapper(MappingConfig())
    target_filter = make_filter()
    state = LoopState()

    returned_filter = handle_neutral_capture(
        sample=None,
        now_ms=100,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=RobotTargets(0.0, 0.0, 50.0),
        min_hand_confidence=0.45,
        stale_timeout_ms=150,
        state=state,
    )

    assert returned_filter is target_filter
    assert not mapper.neutral_ready
    assert state.notice == "neutral rejected: tracking_lost"


def test_neutral_capture_rejects_mapper_validation_failure_with_notice() -> None:
    mapper = PoseMapper(MappingConfig())
    target_filter = make_filter()
    state = LoopState()
    invalid_sample = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(20)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=100,
    )

    returned_filter = handle_neutral_capture(
        sample=invalid_sample,
        now_ms=100,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=RobotTargets(0.0, 0.0, 50.0),
        min_hand_confidence=0.45,
        stale_timeout_ms=150,
        state=state,
    )

    assert returned_filter is target_filter
    assert not mapper.neutral_ready
    assert state.notice == "neutral rejected: HandSample must contain 21 landmarks"


def test_backend_send_exception_disables_sync_and_suppresses_repeated_sends() -> None:
    class FailingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, _targets: RobotTargets) -> dict[str, float]:
            self.calls += 1
            raise RuntimeError("serial write failed")

    backend = FailingBackend()
    target_filter = make_filter()
    state = LoopState(sync_enabled=True)
    active_result = FilterResult(
        RobotTargets(1.0, 1.0, 51.0),
        frozen=False,
        clamped_keys=(),
        reason=FreezeReason.ACTIVE,
    )

    first_result = handle_backend_send(backend, active_result, target_filter, state)
    second_result = handle_backend_send(backend, active_result, target_filter, state)

    assert backend.calls == 1
    assert not state.sync_enabled
    assert state.send_failed
    assert state.notice == "send failed: serial write failed"
    assert first_result.frozen
    assert second_result.frozen
    assert first_result.reason is FreezeReason.PAUSED
    assert second_result.reason is FreezeReason.PAUSED


def test_sync_toggle_stays_disabled_after_send_failure() -> None:
    state = LoopState(sync_enabled=False, notice="send failed: serial write failed", send_failed=True)

    handle_sync_toggle(state)

    assert not state.sync_enabled
    assert state.notice == "sync locked off: send failed"
