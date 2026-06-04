from pathlib import Path

import pytest

from mediapipe_so101.robot_backend import DryRunBackend, SO101Backend, SO101BackendConfig
from mediapipe_so101.types import ACTION_KEYS, RobotTargets


class FakeRobot:
    action_features = {key: float for key in ACTION_KEYS}

    def __init__(
        self,
        observation: dict[str, float] | None = None,
        observation_error: Exception | None = None,
    ) -> None:
        self.connected = False
        self.disconnect_count = 0
        self.observation = observation or {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 60.0,
        }
        self.observation_error = observation_error
        self.sent: list[dict[str, float]] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def get_observation(self) -> dict[str, float]:
        if self.observation_error is not None:
            raise self.observation_error
        return self.observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent.append(action)
        return action


def test_dry_run_backend_never_connects_to_robot() -> None:
    backend = DryRunBackend(default_gripper=70.0)
    backend.connect()

    assert backend.baseline_targets == RobotTargets(0.0, 0.0, 70.0)
    assert backend.last_action is None

    backend.send(RobotTargets(1.0, 2.0, 30.0))

    assert backend.last_action == {
        "wrist_flex.pos": 1.0,
        "wrist_roll.pos": 2.0,
        "gripper.pos": 30.0,
    }


def test_so101_backend_merges_controlled_targets_with_held_startup_pose(tmp_path: Path) -> None:
    robot = FakeRobot()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "arm.json").write_text("{}", encoding="utf-8")
    received_configs: list[SO101BackendConfig] = []

    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="arm",
            calibration_dir=calibration_dir,
            max_relative_target=5.0,
        ),
        robot_factory=lambda config: received_configs.append(config) or robot,
    )
    backend.connect()

    assert received_configs == [backend.config]
    assert backend.baseline_targets == RobotTargets(4.0, 5.0, 60.0)

    backend.send(RobotTargets(10.0, 11.0, 20.0))

    assert robot.sent == [
        {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 11.0,
            "gripper.pos": 20.0,
        }
    ]


def test_so101_backend_requires_calibration_file(tmp_path: Path) -> None:
    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="missing",
            calibration_dir=tmp_path,
            max_relative_target=5.0,
        ),
        robot_factory=lambda _config: FakeRobot(),
    )

    with pytest.raises(FileNotFoundError, match="missing.json"):
        backend.connect()


def test_so101_backend_requires_robot_action_features(tmp_path: Path) -> None:
    robot = FakeRobot()
    robot.action_features = {"wrist_flex.pos": float}
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "arm.json").write_text("{}", encoding="utf-8")

    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="arm",
            calibration_dir=calibration_dir,
            max_relative_target=5.0,
        ),
        robot_factory=lambda _config: robot,
    )

    with pytest.raises(KeyError, match="Robot action features missing keys"):
        backend.connect()

    assert robot.connected is False


def test_so101_backend_rejects_non_finite_target_without_sending(tmp_path: Path) -> None:
    robot = FakeRobot()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "arm.json").write_text("{}", encoding="utf-8")

    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="arm",
            calibration_dir=calibration_dir,
            max_relative_target=5.0,
        ),
        robot_factory=lambda _config: robot,
    )
    backend.connect()

    with pytest.raises(ValueError, match="non-finite"):
        backend.send(RobotTargets(float("nan"), 11.0, 20.0))

    assert robot.sent == []


def test_so101_backend_rejects_non_finite_startup_observation_and_disconnects(
    tmp_path: Path,
) -> None:
    observation = {
        "shoulder_pan.pos": 1.0,
        "shoulder_lift.pos": 2.0,
        "elbow_flex.pos": 3.0,
        "wrist_flex.pos": float("inf"),
        "wrist_roll.pos": 5.0,
        "gripper.pos": 60.0,
    }
    robot = FakeRobot(observation=observation)
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "arm.json").write_text("{}", encoding="utf-8")

    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="arm",
            calibration_dir=calibration_dir,
            max_relative_target=5.0,
        ),
        robot_factory=lambda _config: robot,
    )

    with pytest.raises(ValueError, match="non-finite"):
        backend.connect()

    assert robot.connected is False
    assert robot.disconnect_count == 1


def test_so101_backend_disconnects_if_startup_observation_fails(tmp_path: Path) -> None:
    robot = FakeRobot(observation_error=RuntimeError("camera failed"))
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    (calibration_dir / "arm.json").write_text("{}", encoding="utf-8")

    backend = SO101Backend(
        SO101BackendConfig(
            port="/dev/cu.fake",
            robot_id="arm",
            calibration_dir=calibration_dir,
            max_relative_target=5.0,
        ),
        robot_factory=lambda _config: robot,
    )

    with pytest.raises(RuntimeError, match="camera failed"):
        backend.connect()

    assert robot.connected is False
    assert robot.disconnect_count == 1
