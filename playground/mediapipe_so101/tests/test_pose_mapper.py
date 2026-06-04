import math

import pytest

from mediapipe_so101.pose_mapper import MappingConfig, PoseMapper, extract_features
from mediapipe_so101.types import HandSample, Landmark, RobotTargets


def hand(
    *,
    index_mcp=(0.45, 0.55, 0.0),
    pinky_mcp=(0.55, 0.55, 0.0),
    middle_mcp=(0.50, 0.45, 0.0),
    middle_tip=(0.50, 0.25, 0.0),
    thumb_tip=(0.42, 0.40, 0.0),
    index_tip=(0.58, 0.40, 0.0),
    timestamp_ms=1000,
) -> HandSample:
    points = [Landmark(0.5, 0.65, 0.0) for _ in range(21)]
    points[4] = Landmark(*thumb_tip)
    points[5] = Landmark(*index_mcp)
    points[8] = Landmark(*index_tip)
    points[9] = Landmark(*middle_mcp)
    points[12] = Landmark(*middle_tip)
    points[17] = Landmark(*pinky_mcp)
    return HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=timestamp_ms)


def mapper() -> PoseMapper:
    return PoseMapper(
        MappingConfig(
            wrist_flex_gain=30.0,
            wrist_roll_gain=60.0,
            gripper_open=80.0,
            gripper_closed=20.0,
            pinch_closed_ratio=0.35,
            pinch_open_ratio=1.40,
        )
    )


def test_extract_features_reports_open_pinch() -> None:
    features = extract_features(hand())

    assert features.pinch_open == pytest.approx(1.0)


def test_extract_features_reports_closed_pinch() -> None:
    features = extract_features(hand(thumb_tip=(0.49, 0.40, 0.0), index_tip=(0.51, 0.40, 0.0)))

    assert features.pinch_open == pytest.approx(0.0)


def test_neutral_pose_preserves_wrist_baseline_and_maps_gripper_from_pinch() -> None:
    pose_mapper = mapper()
    neutral = hand()
    pose_mapper.capture_neutral(neutral, RobotTargets(1.0, 2.0, 50.0))

    targets = pose_mapper.map(neutral)

    assert targets.wrist_flex == pytest.approx(1.0)
    assert targets.wrist_roll == pytest.approx(2.0)
    assert targets.gripper == pytest.approx(80.0)


def test_roll_delta_maps_to_expected_wrist_roll() -> None:
    pose_mapper = mapper()
    neutral = hand(index_mcp=(0.45, 0.55, 0.0), pinky_mcp=(0.55, 0.55, 0.0))
    rolled = hand(index_mcp=(0.45, 0.60, 0.0), pinky_mcp=(0.55, 0.50, 0.0))
    pose_mapper.capture_neutral(neutral, RobotTargets(0.0, 0.0, 50.0))

    targets = pose_mapper.map(rolled)

    assert targets.wrist_roll == pytest.approx(math.pi / 4 * 60.0)


def test_flex_delta_maps_to_expected_wrist_flex_with_configured_gain() -> None:
    pose_mapper = PoseMapper(
        MappingConfig(
            wrist_flex_gain=12.0,
            wrist_roll_gain=60.0,
            gripper_open=80.0,
            gripper_closed=20.0,
            pinch_closed_ratio=0.35,
            pinch_open_ratio=1.40,
        )
    )
    neutral = hand(middle_mcp=(0.50, 0.45, 0.0))
    flexed = hand(middle_mcp=(0.50, 0.55, 0.0))
    pose_mapper.capture_neutral(neutral, RobotTargets(0.0, 0.0, 50.0))

    targets = pose_mapper.map(flexed)

    assert targets.wrist_flex == pytest.approx(12.0)


def test_roll_delta_wraps_around_pi_boundary() -> None:
    pose_mapper = mapper()
    neutral = hand(index_mcp=(0.55, 0.505, 0.0), pinky_mcp=(0.45, 0.495, 0.0))
    rolled = hand(index_mcp=(0.55, 0.495, 0.0), pinky_mcp=(0.45, 0.505, 0.0))
    pose_mapper.capture_neutral(neutral, RobotTargets(0.0, 0.0, 50.0))

    targets = pose_mapper.map(rolled)

    expected_delta = math.atan2(-0.01, -0.10) - math.atan2(0.01, -0.10)
    expected_delta = (expected_delta + math.pi) % (2.0 * math.pi) - math.pi
    assert targets.wrist_roll == pytest.approx(expected_delta * 60.0)


def test_full_pinch_maps_to_closed_gripper() -> None:
    pose_mapper = mapper()
    neutral = hand()
    pinched = hand(thumb_tip=(0.49, 0.40, 0.0), index_tip=(0.51, 0.40, 0.0))
    pose_mapper.capture_neutral(neutral, RobotTargets(0.0, 0.0, 50.0))

    targets = pose_mapper.map(pinched)

    assert targets.gripper == pytest.approx(20.0)


def test_map_requires_neutral_capture() -> None:
    pose_mapper = mapper()

    with pytest.raises(RuntimeError, match="Neutral pose has not been captured"):
        pose_mapper.map(hand())


@pytest.mark.parametrize(
    "robot_targets",
    [
        RobotTargets(math.nan, 0.0, 50.0),
        RobotTargets(0.0, math.inf, 50.0),
        RobotTargets(0.0, 0.0, -math.inf),
    ],
)
def test_capture_neutral_rejects_non_finite_robot_targets(robot_targets: RobotTargets) -> None:
    pose_mapper = mapper()

    with pytest.raises(ValueError, match="Robot target values must be finite"):
        pose_mapper.capture_neutral(hand(), robot_targets)


def test_extract_features_rejects_degenerate_hand_width() -> None:
    sample = hand(index_mcp=(0.50, 0.55, 0.0), pinky_mcp=(0.51, 0.55, 0.0))

    with pytest.raises(ValueError, match="Hand width is too small"):
        extract_features(sample)


@pytest.mark.parametrize(
    ("landmark_index", "coordinate"),
    [
        (0, "x"),
        (4, "y"),
        (5, "z"),
        (8, "x"),
        (9, "y"),
        (17, "z"),
    ],
)
def test_extract_features_rejects_non_finite_consumed_landmarks(
    landmark_index: int,
    coordinate: str,
) -> None:
    points = list(hand().landmarks)
    original = points[landmark_index]
    values = {"x": original.x, "y": original.y, "z": original.z}
    values[coordinate] = math.inf
    points[landmark_index] = Landmark(**values)
    sample = HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=1000)

    with pytest.raises(ValueError, match="Landmark coordinates must be finite"):
        extract_features(sample)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wrist_flex_gain": math.inf},
        {"wrist_roll_gain": math.nan},
        {"gripper_open": math.inf},
        {"gripper_closed": -math.inf},
        {"pinch_closed_ratio": math.nan},
        {"pinch_open_ratio": math.inf},
        {"min_hand_width": math.nan},
        {"min_hand_width": 0.0},
        {"min_hand_width": -0.01},
        {"pinch_closed_ratio": 0.4, "pinch_open_ratio": 0.4},
        {"pinch_closed_ratio": 0.5, "pinch_open_ratio": 0.4},
    ],
)
def test_mapping_config_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MappingConfig(**kwargs)
