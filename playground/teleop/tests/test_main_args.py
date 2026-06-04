import math
from pathlib import Path

import pytest

import main


def parsed(*args: str):
    return main.parse_args(args)


def test_default_args_pass_validation() -> None:
    args = parsed()
    main.apply_camera_defaults(args)
    main.validate_args(args)
    assert args.arm == "right"
    assert args.mirror_hand in ("auto", "on", "off")
    assert args.fps > 0


def test_invalid_fps_raises_system_exit() -> None:
    args = parsed("--fps", "0")
    main.apply_camera_defaults(args)
    with pytest.raises(SystemExit, match="--fps"):
        main.validate_args(args)


def test_invalid_confidence_raises_system_exit() -> None:
    args = parsed("--detection-confidence", "1.5")
    main.apply_camera_defaults(args)
    with pytest.raises(SystemExit, match="--detection-confidence"):
        main.validate_args(args)


def test_invalid_min_pose_visibility_raises_system_exit() -> None:
    args = parsed("--min-pose-visibility", "-0.1")
    main.apply_camera_defaults(args)
    with pytest.raises(SystemExit, match="--min-pose-visibility"):
        main.validate_args(args)


def test_arm_choice_is_validated() -> None:
    with pytest.raises(SystemExit):
        parsed("--arm", "middle")


def test_mirror_hand_choice_is_validated() -> None:
    with pytest.raises(SystemExit):
        parsed("--mirror-hand", "maybe")


def test_enable_robot_requires_port(tmp_path: Path) -> None:
    args = parsed("--enable-robot", "--robot-id", "so101_x")
    main.apply_camera_defaults(args)
    args.calibration_dir = tmp_path
    with pytest.raises(SystemExit, match="--robot-port"):
        main.validate_args(args)


def test_enable_robot_requires_id(tmp_path: Path) -> None:
    args = parsed("--enable-robot", "--robot-port", "/dev/null")
    main.apply_camera_defaults(args)
    args.calibration_dir = tmp_path
    with pytest.raises(SystemExit, match="--robot-id"):
        main.validate_args(args)


def test_enable_robot_requires_calibration_file_exists(tmp_path: Path) -> None:
    args = parsed(
        "--enable-robot",
        "--robot-port",
        "/dev/null",
        "--robot-id",
        "so101_x",
    )
    main.apply_camera_defaults(args)
    args.calibration_dir = tmp_path / "missing"
    with pytest.raises(SystemExit, match="Calibration file not found"):
        main.validate_args(args)


def test_gripper_min_must_be_less_than_max() -> None:
    args = parsed("--gripper-min", "80", "--gripper-max", "20")
    main.apply_camera_defaults(args)
    with pytest.raises(SystemExit, match="gripper-min"):
        main.validate_args(args)


def test_robot_port_id_serial_mismatch_raises(tmp_path: Path) -> None:
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "so101_AAAA.json").write_text("{}")
    args = parsed(
        "--enable-robot",
        "--robot-port",
        "/dev/cu.usbmodemBBBB",
        "--robot-id",
        "so101_AAAA",
    )
    main.apply_camera_defaults(args)
    args.calibration_dir = calib
    with pytest.raises(SystemExit, match="appears to be for serial"):
        main.validate_args(args)
