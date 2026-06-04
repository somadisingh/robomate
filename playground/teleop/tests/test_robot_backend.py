import math
from pathlib import Path

import pytest

from teleop.robot_backend import (
    DryRunBackend,
    SO101Backend,
    SO101BackendConfig,
)
from teleop.types import ACTION_KEYS, RobotTargets


def targets(**overrides) -> RobotTargets:
    base = dict(
        shoulder_pan=1.0,
        shoulder_lift=2.0,
        elbow_flex=3.0,
        wrist_flex=4.0,
        wrist_roll=5.0,
        gripper=60.0,
    )
    base.update(overrides)
    return RobotTargets(**base)


def test_dry_run_backend_baseline_uses_zero_arm_and_default_gripper() -> None:
    backend = DryRunBackend(default_gripper=75.0)
    baseline = backend.baseline_targets
    assert baseline.shoulder_pan == 0.0
    assert baseline.shoulder_lift == 0.0
    assert baseline.elbow_flex == 0.0
    assert baseline.wrist_flex == 0.0
    assert baseline.wrist_roll == 0.0
    assert baseline.gripper == 75.0


def test_dry_run_backend_send_returns_action_dict() -> None:
    backend = DryRunBackend(default_gripper=50.0)
    backend.connect()
    action = backend.send(targets())
    assert tuple(action.keys()) == ACTION_KEYS
    assert action["shoulder_pan.pos"] == 1.0
    assert action["gripper.pos"] == 60.0
    backend.disconnect()


def test_dry_run_backend_send_rejects_non_finite() -> None:
    backend = DryRunBackend(default_gripper=50.0)
    backend.connect()
    with pytest.raises(ValueError, match="non-finite"):
        backend.send(targets(shoulder_pan=math.nan))


class FakeRobot:
    def __init__(self, observation: dict[str, float]) -> None:
        self.action_features = {key: float for key in ACTION_KEYS}
        self._observation = observation
        self.connected = False
        self.last_action: dict[str, float] | None = None

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self) -> dict[str, float]:
        return dict(self._observation)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.last_action = action
        return action


def so101_config(calibration_dir: Path) -> SO101BackendConfig:
    return SO101BackendConfig(
        port="/dev/null",
        robot_id="so101_test",
        calibration_dir=calibration_dir,
        max_relative_target=5.0,
    )


def write_calibration(tmp_path: Path) -> Path:
    calibration_dir = tmp_path / "calib"
    calibration_dir.mkdir()
    (calibration_dir / "so101_test.json").write_text("{}")
    return calibration_dir


def test_so101_backend_records_baseline_from_observation_for_all_six_joints(tmp_path: Path) -> None:
    calibration_dir = write_calibration(tmp_path)
    observation = {key: float(idx) for idx, key in enumerate(ACTION_KEYS)}
    robot = FakeRobot(observation)
    backend = SO101Backend(so101_config(calibration_dir), robot_factory=lambda _cfg: robot)

    backend.connect()
    baseline = backend.baseline_targets

    assert baseline.shoulder_pan == 0.0
    assert baseline.shoulder_lift == 1.0
    assert baseline.elbow_flex == 2.0
    assert baseline.wrist_flex == 3.0
    assert baseline.wrist_roll == 4.0
    assert baseline.gripper == 5.0


def test_so101_backend_send_includes_all_six_keys(tmp_path: Path) -> None:
    calibration_dir = write_calibration(tmp_path)
    observation = {key: 0.0 for key in ACTION_KEYS}
    robot = FakeRobot(observation)
    backend = SO101Backend(so101_config(calibration_dir), robot_factory=lambda _cfg: robot)
    backend.connect()

    backend.send(targets())

    assert robot.last_action is not None
    assert set(robot.last_action.keys()) == set(ACTION_KEYS)
    assert robot.last_action["shoulder_pan.pos"] == 1.0
    assert robot.last_action["gripper.pos"] == 60.0


def test_so101_backend_send_before_connect_raises(tmp_path: Path) -> None:
    calibration_dir = write_calibration(tmp_path)
    backend = SO101Backend(so101_config(calibration_dir), robot_factory=lambda _cfg: FakeRobot({}))
    with pytest.raises(RuntimeError, match="not connected"):
        backend.send(targets())


def test_so101_backend_missing_calibration_raises(tmp_path: Path) -> None:
    backend = SO101Backend(
        so101_config(tmp_path / "missing"),
        robot_factory=lambda _cfg: FakeRobot({}),
    )
    with pytest.raises(FileNotFoundError, match="calibration"):
        backend.connect()


def test_so101_backend_missing_action_features_raises(tmp_path: Path) -> None:
    calibration_dir = write_calibration(tmp_path)
    bad_robot = FakeRobot({key: 0.0 for key in ACTION_KEYS})
    bad_robot.action_features = {"shoulder_pan.pos": float}
    backend = SO101Backend(so101_config(calibration_dir), robot_factory=lambda _cfg: bad_robot)
    with pytest.raises(KeyError, match="action features"):
        backend.connect()
