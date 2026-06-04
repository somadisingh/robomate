import pytest

from mediapipe_so101.safety import SafetyConfig, TargetFilter
from mediapipe_so101.types import FreezeReason, RobotTargets


def make_config(
    *,
    limits: dict[str, tuple[float, float]] | None = None,
    max_delta: dict[str, float] | None = None,
    smoothing: float = 1.0,
    stale_timeout_ms: int = 150,
) -> SafetyConfig:
    return SafetyConfig(
        limits=limits
        or {
            "wrist_flex.pos": (-20.0, 20.0),
            "wrist_roll.pos": (-30.0, 30.0),
            "gripper.pos": (20.0, 80.0),
        },
        max_delta=max_delta
        or {
            "wrist_flex.pos": 5.0,
            "wrist_roll.pos": 10.0,
            "gripper.pos": 15.0,
        },
        smoothing=smoothing,
        stale_timeout_ms=stale_timeout_ms,
    )


def make_filter() -> TargetFilter:
    return TargetFilter(
        make_config(),
        initial_targets=RobotTargets(0.0, 0.0, 50.0),
    )


def test_limits_are_clamped() -> None:
    filt = make_filter()

    result = filt.update(
        RobotTargets(wrist_flex=100.0, wrist_roll=-100.0, gripper=100.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(5.0, -10.0, 65.0)
    assert result.clamped_keys == ("gripper.pos", "wrist_flex.pos", "wrist_roll.pos")
    assert result.frozen is False
    assert result.reason is FreezeReason.ACTIVE


def test_freezes_when_paused() -> None:
    filt = make_filter()

    result = filt.update(
        RobotTargets(wrist_flex=10.0, wrist_roll=10.0, gripper=80.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=False,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(0.0, 0.0, 50.0)
    assert result.frozen is True
    assert result.reason is FreezeReason.PAUSED


def test_freezes_when_neutral_missing() -> None:
    filt = make_filter()

    result = filt.update(
        RobotTargets(wrist_flex=10.0, wrist_roll=10.0, gripper=80.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=False,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(0.0, 0.0, 50.0)
    assert result.reason is FreezeReason.NEUTRAL_MISSING


def test_freezes_stale_tracking() -> None:
    filt = make_filter()

    result = filt.update(
        RobotTargets(wrist_flex=10.0, wrist_roll=10.0, gripper=80.0),
        now_ms=1200,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(0.0, 0.0, 50.0)
    assert result.reason is FreezeReason.STALE_RESULT


def test_smoothing_blends_from_previous_target() -> None:
    filt = TargetFilter(
        SafetyConfig(
            limits={
                "wrist_flex.pos": (-100.0, 100.0),
                "wrist_roll.pos": (-100.0, 100.0),
                "gripper.pos": (0.0, 100.0),
            },
            max_delta={
                "wrist_flex.pos": 100.0,
                "wrist_roll.pos": 100.0,
                "gripper.pos": 100.0,
            },
            smoothing=0.25,
            stale_timeout_ms=150,
        ),
        initial_targets=RobotTargets(0.0, 0.0, 0.0),
    )

    result = filt.update(
        RobotTargets(wrist_flex=40.0, wrist_roll=80.0, gripper=100.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(10.0, 20.0, 25.0)


def test_rejects_missing_limit_key() -> None:
    with pytest.raises(ValueError, match="Missing safety limits"):
        make_config(
            limits={
                "wrist_flex.pos": (-20.0, 20.0),
                "wrist_roll.pos": (-30.0, 30.0),
            },
        )


def test_rejects_missing_max_delta_key() -> None:
    with pytest.raises(ValueError, match="Missing max_delta values"):
        make_config(
            max_delta={
                "wrist_flex.pos": 5.0,
                "wrist_roll.pos": 10.0,
            },
        )


@pytest.mark.parametrize("smoothing", [0.0, -0.1, 1.1])
def test_rejects_invalid_smoothing(smoothing: float) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        make_config(smoothing=smoothing)


@pytest.mark.parametrize(
    "stale_timeout_ms",
    [float("inf"), float("nan"), 150.0, True, "150", 0, -1],
)
def test_rejects_invalid_stale_timeout(stale_timeout_ms: object) -> None:
    with pytest.raises(ValueError, match="stale_timeout_ms"):
        make_config(stale_timeout_ms=stale_timeout_ms)


@pytest.mark.parametrize("max_delta", [0.0, -1.0])
def test_rejects_non_positive_max_delta(max_delta: float) -> None:
    with pytest.raises(ValueError, match="max_delta"):
        make_config(
            max_delta={
                "wrist_flex.pos": max_delta,
                "wrist_roll.pos": 10.0,
                "gripper.pos": 15.0,
            },
        )


@pytest.mark.parametrize("limit", [(20.0, 20.0), (20.0, -20.0)])
def test_rejects_inverted_or_equal_hard_limits(
    limit: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="limit"):
        make_config(
            limits={
                "wrist_flex.pos": limit,
                "wrist_roll.pos": (-30.0, 30.0),
                "gripper.pos": (20.0, 80.0),
            },
        )


def test_rejects_initial_target_outside_hard_limits() -> None:
    with pytest.raises(ValueError, match="initial target"):
        TargetFilter(
            make_config(),
            initial_targets=RobotTargets(wrist_flex=25.0, wrist_roll=0.0, gripper=50.0),
        )


def test_hard_limit_clamp_applies_when_delta_allows_desired_target() -> None:
    filt = TargetFilter(
        SafetyConfig(
            limits={
                "wrist_flex.pos": (-20.0, 20.0),
                "wrist_roll.pos": (-30.0, 30.0),
                "gripper.pos": (20.0, 80.0),
            },
            max_delta={
                "wrist_flex.pos": 500.0,
                "wrist_roll.pos": 500.0,
                "gripper.pos": 500.0,
            },
            smoothing=1.0,
            stale_timeout_ms=150,
        ),
        initial_targets=RobotTargets(0.0, 0.0, 50.0),
    )

    result = filt.update(
        RobotTargets(wrist_flex=100.0, wrist_roll=-100.0, gripper=100.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(20.0, -30.0, 80.0)
    assert result.clamped_keys == ("gripper.pos", "wrist_flex.pos", "wrist_roll.pos")


@pytest.mark.parametrize("limit", [(float("nan"), 20.0), (-20.0, float("inf"))])
def test_rejects_non_finite_hard_limits(limit: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_config(
            limits={
                "wrist_flex.pos": limit,
                "wrist_roll.pos": (-30.0, 30.0),
                "gripper.pos": (20.0, 80.0),
            },
        )


@pytest.mark.parametrize("max_delta", [float("nan"), float("inf")])
def test_rejects_non_finite_max_delta(max_delta: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_config(
            max_delta={
                "wrist_flex.pos": max_delta,
                "wrist_roll.pos": 10.0,
                "gripper.pos": 15.0,
            },
        )


@pytest.mark.parametrize("smoothing", [float("nan"), float("inf")])
def test_rejects_non_finite_smoothing(smoothing: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_config(smoothing=smoothing)


@pytest.mark.parametrize(
    "initial_targets",
    [
        RobotTargets(wrist_flex=float("nan"), wrist_roll=0.0, gripper=50.0),
        RobotTargets(wrist_flex=0.0, wrist_roll=float("inf"), gripper=50.0),
        RobotTargets(wrist_flex=0.0, wrist_roll=0.0, gripper=float("-inf")),
    ],
)
def test_rejects_non_finite_initial_targets(initial_targets: RobotTargets) -> None:
    with pytest.raises(ValueError, match="finite"):
        TargetFilter(make_config(), initial_targets=initial_targets)


@pytest.mark.parametrize(
    "desired",
    [
        RobotTargets(wrist_flex=float("nan"), wrist_roll=10.0, gripper=60.0),
        RobotTargets(wrist_flex=10.0, wrist_roll=float("inf"), gripper=60.0),
        RobotTargets(wrist_flex=10.0, wrist_roll=10.0, gripper=float("-inf")),
    ],
)
def test_non_finite_desired_target_freezes_at_last_safe_target(
    desired: RobotTargets,
) -> None:
    filt = make_filter()

    result = filt.update(
        RobotTargets(wrist_flex=5.0, wrist_roll=5.0, gripper=55.0),
        now_ms=1000,
        sample_timestamp_ms=1000,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )
    assert result.targets == RobotTargets(5.0, 5.0, 55.0)

    result = filt.update(
        desired,
        now_ms=1010,
        sample_timestamp_ms=1010,
        sync_enabled=True,
        neutral_ready=True,
        deadman_active=True,
        tracking_ok=True,
    )

    assert result.targets == RobotTargets(5.0, 5.0, 55.0)
    assert result.frozen is True
    assert result.clamped_keys == ()
    assert result.reason is FreezeReason.TRACKING_LOST
    assert filt.last_targets == RobotTargets(5.0, 5.0, 55.0)


def test_safety_config_defensively_copies_input_dicts() -> None:
    limits = {
        "wrist_flex.pos": (-20.0, 20.0),
        "wrist_roll.pos": (-30.0, 30.0),
        "gripper.pos": (20.0, 80.0),
    }
    max_delta = {
        "wrist_flex.pos": 5.0,
        "wrist_roll.pos": 10.0,
        "gripper.pos": 15.0,
    }

    config = SafetyConfig(
        limits=limits,
        max_delta=max_delta,
        smoothing=1.0,
        stale_timeout_ms=150,
    )
    limits["wrist_flex.pos"] = (-1000.0, 1000.0)
    max_delta["wrist_flex.pos"] = 1000.0

    assert config.limits["wrist_flex.pos"] == (-20.0, 20.0)
    assert config.max_delta["wrist_flex.pos"] == 5.0


def test_safety_config_normalizes_list_limits_before_freezing() -> None:
    wrist_limits = [-20.0, 20.0]
    limits = {
        "wrist_flex.pos": wrist_limits,
        "wrist_roll.pos": [-30.0, 30.0],
        "gripper.pos": [20.0, 80.0],
    }

    config = SafetyConfig(
        limits=limits,
        max_delta={
            "wrist_flex.pos": 5.0,
            "wrist_roll.pos": 10.0,
            "gripper.pos": 15.0,
        },
        smoothing=1.0,
        stale_timeout_ms=150,
    )
    wrist_limits[0] = -1000.0
    wrist_limits[1] = 1000.0

    assert config.limits["wrist_flex.pos"] == (-20.0, 20.0)


def test_safety_config_mappings_are_immutable() -> None:
    config = make_config()

    with pytest.raises(TypeError):
        config.limits["wrist_flex.pos"] = (-1000.0, 1000.0)
    with pytest.raises(TypeError):
        config.max_delta["wrist_flex.pos"] = 1000.0
