from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .types import ACTION_KEYS, RobotTargets


class RobotLike(Protocol):
    action_features: dict[str, type]

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, float]: ...
    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...


class Backend(Protocol):
    @property
    def baseline_targets(self) -> RobotTargets: ...
    def connect(self) -> None: ...
    def send(self, targets: RobotTargets) -> dict[str, float]: ...
    def disconnect(self) -> None: ...


class DryRunBackend:
    def __init__(self, default_gripper: float) -> None:
        self._baseline = RobotTargets(
            shoulder_pan=0.0,
            shoulder_lift=0.0,
            elbow_flex=0.0,
            wrist_flex=0.0,
            wrist_roll=0.0,
            gripper=default_gripper,
        )
        self.last_action: dict[str, float] | None = None

    @property
    def baseline_targets(self) -> RobotTargets:
        return self._baseline

    def connect(self) -> None:
        return None

    def send(self, targets: RobotTargets) -> dict[str, float]:
        _validate_targets_are_finite(targets)
        self.last_action = targets.as_action()
        return self.last_action

    def disconnect(self) -> None:
        return None


@dataclass(frozen=True)
class SO101BackendConfig:
    port: str
    robot_id: str
    calibration_dir: Path
    max_relative_target: float


class SO101Backend:
    def __init__(
        self,
        config: SO101BackendConfig,
        robot_factory: Callable[[SO101BackendConfig], RobotLike] | None = None,
    ) -> None:
        self.config = config
        self._robot_factory = robot_factory
        self._robot: RobotLike | None = None
        self._baseline = RobotTargets(0.0, 0.0, 0.0, 0.0, 0.0, 50.0)

    @property
    def baseline_targets(self) -> RobotTargets:
        return self._baseline

    def connect(self) -> None:
        calibration_file = self.config.calibration_dir / f"{self.config.robot_id}.json"
        if not calibration_file.exists():
            raise FileNotFoundError(f"SO101 calibration file not found: {calibration_file}")

        robot = (
            self._robot_factory(self.config)
            if self._robot_factory is not None
            else _make_so101_robot(_make_so101_config(self.config))
        )
        _validate_action_features(robot)
        robot.connect()
        try:
            observation = robot.get_observation()
            action = _extract_action(observation)
            self._baseline = RobotTargets(
                shoulder_pan=action["shoulder_pan.pos"],
                shoulder_lift=action["shoulder_lift.pos"],
                elbow_flex=action["elbow_flex.pos"],
                wrist_flex=action["wrist_flex.pos"],
                wrist_roll=action["wrist_roll.pos"],
                gripper=action["gripper.pos"],
            )
            self._robot = robot
        except Exception:
            robot.disconnect()
            raise

    def send(self, targets: RobotTargets) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("SO101Backend is not connected")
        _validate_targets_are_finite(targets)
        return self._robot.send_action(targets.as_action())

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None


def _extract_action(observation: dict[str, float]) -> dict[str, float]:
    missing = set(ACTION_KEYS) - set(observation)
    if missing:
        raise KeyError(f"Robot observation missing action keys: {sorted(missing)}")
    action = {key: float(observation[key]) for key in ACTION_KEYS}
    _validate_action_values_are_finite(action)
    return action


def _validate_targets_are_finite(targets: RobotTargets) -> None:
    _validate_action_values_are_finite(targets.as_action())


def _validate_action_values_are_finite(action: dict[str, float]) -> None:
    non_finite = [key for key, value in action.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"Robot action contains non-finite values: {non_finite}")


def _validate_action_features(robot: RobotLike) -> None:
    missing = set(ACTION_KEYS) - set(robot.action_features)
    if missing:
        raise KeyError(f"Robot action features missing keys: {sorted(missing)}")


def _make_so101_config(config: SO101BackendConfig) -> object:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    return SOFollowerRobotConfig(
        port=config.port,
        id=config.robot_id,
        calibration_dir=config.calibration_dir,
        max_relative_target=config.max_relative_target,
        use_degrees=True,
    )


def _make_so101_robot(config: object) -> RobotLike:
    from lerobot.robots.so_follower.so_follower import SO101Follower

    return SO101Follower(config)
