import math

import pytest

from teleop.safety import SafetyConfig, TargetFilter
from teleop.types import (
    CONTROLLED_KEYS,
    FreezeReason,
    RobotTargets,
)


def baseline() -> RobotTargets:
    return RobotTargets(
        shoulder_pan=0.0,
        shoulder_lift=0.0,
        elbow_flex=0.0,
        wrist_flex=0.0,
        wrist_roll=0.0,
        gripper=50.0,
    )


def safety_config(**overrides) -> SafetyConfig:
    defaults = dict(
        limits={
            "shoulder_pan.pos": (-30.0, 30.0),
            "shoulder_lift.pos": (-30.0, 30.0),
            "elbow_flex.pos": (-30.0, 30.0),
            "wrist_flex.pos": (-30.0, 30.0),
            "wrist_roll.pos": (-30.0, 30.0),
            "gripper.pos": (0.0, 100.0),
        },
        max_delta={key: 5.0 for key in CONTROLLED_KEYS},
        smoothing=1.0,
        stale_timeout_ms=150,
        min_pose_visibility=0.5,
        min_hand_confidence=0.5,
    )
    defaults.update(overrides)
    return SafetyConfig(**defaults)


def filter_(config: SafetyConfig | None = None, initial: RobotTargets | None = None) -> TargetFilter:
    return TargetFilter(config or safety_config(), initial or baseline())


def update_kwargs(**overrides) -> dict:
    defaults = dict(
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )
    defaults.update(overrides)
    return defaults


def test_safety_config_requires_limits_for_all_six_keys() -> None:
    with pytest.raises(ValueError, match="Missing safety limits"):
        SafetyConfig(
            limits={"shoulder_pan.pos": (-1.0, 1.0)},
            max_delta={key: 1.0 for key in CONTROLLED_KEYS},
            smoothing=0.5,
            stale_timeout_ms=100,
            min_pose_visibility=0.5,
            min_hand_confidence=0.5,
        )


def test_safety_config_requires_max_delta_for_all_six_keys() -> None:
    with pytest.raises(ValueError, match="Missing max_delta"):
        SafetyConfig(
            limits={key: (-1.0, 1.0) for key in CONTROLLED_KEYS},
            max_delta={"shoulder_pan.pos": 1.0},
            smoothing=0.5,
            stale_timeout_ms=100,
            min_pose_visibility=0.5,
            min_hand_confidence=0.5,
        )


def test_initial_targets_outside_limits_raise() -> None:
    out_of_range = RobotTargets(
        shoulder_pan=100.0,
        shoulder_lift=0.0,
        elbow_flex=0.0,
        wrist_flex=0.0,
        wrist_roll=0.0,
        gripper=50.0,
    )
    with pytest.raises(ValueError, match="outside safety limit"):
        TargetFilter(safety_config(), out_of_range)


def test_filter_freezes_when_sync_disabled() -> None:
    f = filter_()
    result = f.update(
        RobotTargets(1.0, 0.0, 0.0, 0.0, 0.0, 50.0), **update_kwargs(sync_enabled=False)
    )
    assert result.frozen is True
    assert result.reason is FreezeReason.PAUSED
    assert result.targets == baseline()


def test_filter_freezes_when_neutral_not_ready() -> None:
    f = filter_()
    result = f.update(
        RobotTargets(1.0, 0.0, 0.0, 0.0, 0.0, 50.0), **update_kwargs(neutral_ready=False)
    )
    assert result.frozen is True
    assert result.reason is FreezeReason.NEUTRAL_MISSING


def test_filter_freezes_when_tracking_lost() -> None:
    f = filter_()
    result = f.update(
        RobotTargets(1.0, 0.0, 0.0, 0.0, 0.0, 50.0), **update_kwargs(tracking_ok=False)
    )
    assert result.frozen is True
    assert result.reason is FreezeReason.TRACKING_LOST


def test_filter_freezes_when_sample_is_stale() -> None:
    f = filter_()
    result = f.update(
        RobotTargets(1.0, 0.0, 0.0, 0.0, 0.0, 50.0),
        **update_kwargs(now_ms=2000, sample_timestamp_ms=1000),
    )
    assert result.frozen is True
    assert result.reason is FreezeReason.STALE_RESULT


def test_filter_freezes_when_targets_contain_non_finite() -> None:
    f = filter_()
    bad = RobotTargets(math.nan, 0.0, 0.0, 0.0, 0.0, 50.0)
    result = f.update(bad, **update_kwargs())
    assert result.frozen is True
    assert result.reason is FreezeReason.TRACKING_LOST


def test_filter_clamps_each_joint_independently_to_its_limit() -> None:
    f = filter_(config=safety_config(max_delta={key: 1000.0 for key in CONTROLLED_KEYS}))
    desired = RobotTargets(
        shoulder_pan=500.0,
        shoulder_lift=-500.0,
        elbow_flex=5.0,
        wrist_flex=5.0,
        wrist_roll=5.0,
        gripper=200.0,
    )
    result = f.update(desired, **update_kwargs())
    assert result.targets.shoulder_pan == pytest.approx(30.0)
    assert result.targets.shoulder_lift == pytest.approx(-30.0)
    assert result.targets.elbow_flex == pytest.approx(5.0)
    assert result.targets.gripper == pytest.approx(100.0)
    assert "shoulder_pan.pos" in result.clamped_keys
    assert "shoulder_lift.pos" in result.clamped_keys
    assert "gripper.pos" in result.clamped_keys
    assert "elbow_flex.pos" not in result.clamped_keys


def test_filter_rate_limits_each_joint_independently() -> None:
    f = filter_(config=safety_config(max_delta={key: 1.0 for key in CONTROLLED_KEYS}))
    result = f.update(
        RobotTargets(10.0, 10.0, 10.0, 10.0, 10.0, 70.0), **update_kwargs()
    )
    assert result.targets.shoulder_pan == pytest.approx(1.0)
    assert result.targets.shoulder_lift == pytest.approx(1.0)
    assert result.targets.elbow_flex == pytest.approx(1.0)
    assert result.targets.wrist_flex == pytest.approx(1.0)
    assert result.targets.wrist_roll == pytest.approx(1.0)
    assert result.targets.gripper == pytest.approx(51.0)


def test_filter_smooths_with_alpha() -> None:
    f = filter_(config=safety_config(smoothing=0.5, max_delta={key: 100.0 for key in CONTROLLED_KEYS}))
    result = f.update(
        RobotTargets(10.0, 10.0, 10.0, 10.0, 10.0, 60.0), **update_kwargs()
    )
    assert result.targets.shoulder_pan == pytest.approx(5.0)
    assert result.targets.gripper == pytest.approx(55.0)


def test_filter_active_result_updates_last_targets() -> None:
    f = filter_()
    desired = RobotTargets(2.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    result = f.update(desired, **update_kwargs())
    assert result.reason is FreezeReason.ACTIVE
    assert result.frozen is False
    assert f.last_targets.shoulder_pan == pytest.approx(2.0)
