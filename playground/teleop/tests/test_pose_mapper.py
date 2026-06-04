import math

import pytest

from teleop.pose_mapper import (
    MappingConfig,
    WristMapper,
    extract_wrist_features,
)
from teleop.types import HandSample, Landmark, RobotTargets


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


def baseline_targets() -> RobotTargets:
    return RobotTargets(
        shoulder_pan=0.0,
        shoulder_lift=0.0,
        elbow_flex=0.0,
        wrist_flex=1.0,
        wrist_roll=2.0,
        gripper=50.0,
    )


def wrist_mapper(**overrides) -> WristMapper:
    defaults = dict(
        shoulder_pan_gain=30.0,
        shoulder_lift_gain=30.0,
        elbow_flex_gain=30.0,
        wrist_flex_gain=30.0,
        wrist_roll_gain=60.0,
        gripper_open=80.0,
        gripper_closed=20.0,
        pinch_closed_ratio=0.35,
        pinch_open_ratio=1.40,
        mirror_hand=False,
    )
    defaults.update(overrides)
    config = MappingConfig(**defaults)
    return WristMapper(config)


def test_extract_wrist_features_reports_open_pinch() -> None:
    features = extract_wrist_features(hand())
    assert features.pinch_open == pytest.approx(1.0)


def test_extract_wrist_features_reports_closed_pinch() -> None:
    features = extract_wrist_features(
        hand(thumb_tip=(0.49, 0.40, 0.0), index_tip=(0.51, 0.40, 0.0))
    )
    assert features.pinch_open == pytest.approx(0.0)


def test_neutral_capture_preserves_wrist_baseline_and_maps_gripper_from_pinch() -> None:
    mapper = wrist_mapper()
    neutral = hand()
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(neutral)

    assert targets.wrist_flex == pytest.approx(1.0)
    assert targets.wrist_roll == pytest.approx(2.0)
    assert targets.gripper == pytest.approx(80.0)


def test_roll_delta_maps_with_configured_gain() -> None:
    mapper = wrist_mapper()
    neutral = hand(index_mcp=(0.45, 0.55, 0.0), pinky_mcp=(0.55, 0.55, 0.0))
    rolled = hand(index_mcp=(0.45, 0.60, 0.0), pinky_mcp=(0.55, 0.50, 0.0))
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(rolled)

    assert targets.wrist_roll == pytest.approx(2.0 + math.pi / 4 * 60.0)


def test_flex_delta_maps_with_configured_gain() -> None:
    mapper = wrist_mapper(wrist_flex_gain=12.0)
    neutral = hand(middle_mcp=(0.50, 0.45, 0.0))
    flexed = hand(middle_mcp=(0.50, 0.55, 0.0))
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(flexed)

    assert targets.wrist_flex == pytest.approx(1.0 + 12.0)


def test_full_pinch_maps_to_closed_gripper() -> None:
    mapper = wrist_mapper()
    neutral = hand()
    pinched = hand(thumb_tip=(0.49, 0.40, 0.0), index_tip=(0.51, 0.40, 0.0))
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(pinched)

    assert targets.gripper == pytest.approx(20.0)


def test_mirror_hand_inverts_roll_sign() -> None:
    mapper = wrist_mapper(mirror_hand=True)
    neutral = hand(index_mcp=(0.45, 0.55, 0.0), pinky_mcp=(0.55, 0.55, 0.0))
    rolled = hand(index_mcp=(0.45, 0.60, 0.0), pinky_mcp=(0.55, 0.50, 0.0))
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(rolled)

    assert targets.wrist_roll == pytest.approx(2.0 - math.pi / 4 * 60.0)


def test_map_requires_neutral_capture() -> None:
    mapper = wrist_mapper()
    with pytest.raises(RuntimeError, match="Neutral .* has not been captured"):
        mapper.map(hand())


def test_extract_wrist_features_rejects_degenerate_hand_width() -> None:
    sample = hand(index_mcp=(0.50, 0.55, 0.0), pinky_mcp=(0.51, 0.55, 0.0))
    with pytest.raises(ValueError, match="Hand width is too small"):
        extract_wrist_features(sample)


from teleop.pose_mapper import ArmMapper, extract_arm_features
from teleop.types import ArmSample, PoseLandmark


def arm(
    *,
    shoulder=(0.0, 0.0, 0.0),
    elbow=(0.0, 0.3, 0.0),
    wrist=(0.0, 0.6, 0.0),
    visibility=0.9,
    shoulder_image_xy=(0.2, 0.2),
    elbow_image_xy=(0.4, 0.4),
    wrist_image_xy=(0.5, 0.5),
    timestamp_ms=1000,
) -> ArmSample:
    return ArmSample(
        shoulder=PoseLandmark(*shoulder, visibility=visibility),
        shoulder_image_xy=shoulder_image_xy,
        elbow=PoseLandmark(*elbow, visibility=visibility),
        elbow_image_xy=elbow_image_xy,
        wrist=PoseLandmark(*wrist, visibility=visibility),
        wrist_image_xy=wrist_image_xy,
        timestamp_ms=timestamp_ms,
    )


def arm_mapper(**overrides) -> ArmMapper:
    defaults = dict(
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
    defaults.update(overrides)
    return ArmMapper(MappingConfig(**defaults))


def test_arm_hanging_straight_down_yields_zero_elbow_flex_and_shoulder_lift() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.3, 0.0),
        wrist=(0.0, 0.6, 0.0),
    )

    features = extract_arm_features(sample, MappingConfig())

    assert features.elbow_flex == pytest.approx(0.0, abs=1e-6)
    assert features.shoulder_lift == pytest.approx(0.0, abs=1e-6)


def test_arm_horizontal_forward_yields_pi_over_two_shoulder_lift() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.0, 0.3),
        wrist=(0.0, 0.0, 0.6),
    )

    features = extract_arm_features(sample, MappingConfig())

    assert features.shoulder_lift == pytest.approx(math.pi / 2, abs=1e-6)
    assert features.shoulder_pan == pytest.approx(0.0, abs=1e-6)


def test_arm_horizontal_to_right_yields_shoulder_pan_pi_over_two() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.3, 0.0, 0.0),
        wrist=(0.6, 0.0, 0.0),
    )

    features = extract_arm_features(sample, MappingConfig())

    assert features.shoulder_pan == pytest.approx(math.pi / 2, abs=1e-6)


def test_arm_bent_ninety_degrees_yields_pi_over_two_elbow_flex() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.3, 0.0),
        wrist=(0.0, 0.3, 0.3),
    )

    features = extract_arm_features(sample, MappingConfig())

    assert features.elbow_flex == pytest.approx(math.pi / 2, abs=1e-6)


def test_arm_fully_folded_yields_pi_elbow_flex() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.3, 0.0),
        wrist=(0.0, 0.0, 0.0),
    )

    features = extract_arm_features(sample, MappingConfig())

    assert features.elbow_flex == pytest.approx(math.pi, abs=1e-6)


def test_arm_features_invariant_to_whole_body_translation() -> None:
    base = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.1, 0.3, 0.2),
        wrist=(0.15, 0.5, 0.3),
    )
    translated = arm(
        shoulder=(5.0, -1.0, 2.0),
        elbow=(5.1, -0.7, 2.2),
        wrist=(5.15, -0.5, 2.3),
    )

    base_features = extract_arm_features(base, MappingConfig())
    translated_features = extract_arm_features(translated, MappingConfig())

    assert base_features.shoulder_pan == pytest.approx(translated_features.shoulder_pan)
    assert base_features.shoulder_lift == pytest.approx(translated_features.shoulder_lift)
    assert base_features.elbow_flex == pytest.approx(translated_features.elbow_flex)


def test_arm_mapper_neutral_capture_yields_baseline_when_remapped() -> None:
    mapper = arm_mapper()
    neutral = arm()
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(neutral)

    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan)
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift)
    assert targets.elbow_flex == pytest.approx(baseline_targets().elbow_flex)


def test_arm_mapper_emits_delta_from_neutral_scaled_by_gain() -> None:
    mapper = arm_mapper(shoulder_lift_gain=2.0)
    neutral = arm(elbow=(0.0, 0.3, 0.0), wrist=(0.0, 0.6, 0.0))
    moved = arm(elbow=(0.0, 0.0, 0.3), wrist=(0.0, 0.0, 0.6))
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(moved)

    expected_delta = math.pi / 2 - 0.0
    assert targets.shoulder_lift == pytest.approx(
        baseline_targets().shoulder_lift + expected_delta * 2.0
    )


def test_arm_mapper_map_requires_neutral_capture() -> None:
    mapper = arm_mapper()
    with pytest.raises(RuntimeError, match="Neutral arm features have not been captured"):
        mapper.map(arm())


def test_arm_features_rejects_degenerate_upper_arm() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.001, 0.0),
        wrist=(0.0, 0.5, 0.0),
    )
    with pytest.raises(ValueError, match="Upper arm segment is too short"):
        extract_arm_features(sample, MappingConfig())


def test_arm_features_rejects_degenerate_forearm() -> None:
    sample = arm(
        shoulder=(0.0, 0.0, 0.0),
        elbow=(0.0, 0.3, 0.0),
        wrist=(0.0, 0.301, 0.0),
    )
    with pytest.raises(ValueError, match="Forearm segment is too short"):
        extract_arm_features(sample, MappingConfig())


from teleop.pose_mapper import TeleopMapper
from teleop.types import TeleopSample


def teleop_sample(*, arm_sample=None, hand_sample=None, timestamp_ms=1000) -> TeleopSample:
    return TeleopSample(
        arm=arm_sample if arm_sample is not None else arm(),
        hand=hand_sample if hand_sample is not None else hand(),
        timestamp_ms=timestamp_ms,
    )


def teleop_mapper(**overrides) -> TeleopMapper:
    defaults = dict(
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
    defaults.update(overrides)
    return TeleopMapper(MappingConfig(**defaults))


def test_teleop_mapper_neutral_requires_both_arm_and_hand() -> None:
    mapper = teleop_mapper()

    with pytest.raises(ValueError, match="requires both arm and hand"):
        mapper.capture_neutral(
            TeleopSample(arm=None, hand=hand(), timestamp_ms=1000), baseline_targets()
        )
    with pytest.raises(ValueError, match="requires both arm and hand"):
        mapper.capture_neutral(
            TeleopSample(arm=arm(), hand=None, timestamp_ms=1000), baseline_targets()
        )


def test_teleop_mapper_emits_six_dof_targets_after_neutral_capture() -> None:
    mapper = teleop_mapper()
    sample = teleop_sample()
    mapper.capture_neutral(sample, baseline_targets())

    targets = mapper.map(sample)

    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan, abs=1e-6)
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift, abs=1e-6)
    assert targets.elbow_flex == pytest.approx(baseline_targets().elbow_flex, abs=1e-6)
    assert targets.wrist_flex == pytest.approx(baseline_targets().wrist_flex)
    assert targets.wrist_roll == pytest.approx(baseline_targets().wrist_roll)
    assert targets.gripper == pytest.approx(80.0)


def test_teleop_mapper_neutral_ready_requires_both_sides() -> None:
    mapper = teleop_mapper()
    assert mapper.neutral_ready is False

    mapper.capture_neutral(teleop_sample(), baseline_targets())
    assert mapper.neutral_ready is True


def test_teleop_mapper_map_requires_arm_and_hand_present() -> None:
    mapper = teleop_mapper()
    mapper.capture_neutral(teleop_sample(), baseline_targets())

    with pytest.raises(ValueError, match="requires both arm and hand"):
        mapper.map(TeleopSample(arm=None, hand=hand(), timestamp_ms=1000))


def test_teleop_mapper_map_requires_neutral_capture() -> None:
    mapper = teleop_mapper()
    with pytest.raises(RuntimeError, match="Neutral .* has not been captured"):
        mapper.map(teleop_sample())
