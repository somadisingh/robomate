from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .types import CONTROLLED_KEYS, FilterResult, FreezeReason, RobotTargets


@dataclass(frozen=True)
class SafetyConfig:
    limits: Mapping[str, tuple[float, float]]
    max_delta: Mapping[str, float]
    smoothing: float
    stale_timeout_ms: int
    min_pose_visibility: float
    min_hand_confidence: float

    def __post_init__(self) -> None:
        missing_limits = set(CONTROLLED_KEYS) - set(self.limits)
        missing_delta = set(CONTROLLED_KEYS) - set(self.max_delta)
        if missing_limits:
            raise ValueError(f"Missing safety limits for {sorted(missing_limits)}")
        if missing_delta:
            raise ValueError(f"Missing max_delta values for {sorted(missing_delta)}")

        validated_limits: dict[str, tuple[float, float]] = {}
        validated_max_delta: dict[str, float] = {}
        for key in CONTROLLED_KEYS:
            low, high = self.limits[key]
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"Safety limits for {key} must be finite")
            if low >= high:
                raise ValueError(f"Invalid safety limit for {key}: {low} >= {high}")
            validated_limits[key] = (low, high)

            max_delta = self.max_delta[key]
            if not math.isfinite(max_delta):
                raise ValueError(f"max_delta for {key} must be finite")
            if max_delta <= 0:
                raise ValueError(f"Invalid max_delta for {key}: {max_delta} <= 0")
            validated_max_delta[key] = max_delta

        if not math.isfinite(self.smoothing):
            raise ValueError("smoothing must be finite")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in the interval (0, 1]")
        if (
            not isinstance(self.stale_timeout_ms, int)
            or isinstance(self.stale_timeout_ms, bool)
            or self.stale_timeout_ms <= 0
        ):
            raise ValueError("stale_timeout_ms must be a positive integer")
        if not math.isfinite(self.min_pose_visibility) or not 0.0 <= self.min_pose_visibility <= 1.0:
            raise ValueError("min_pose_visibility must be in [0, 1]")
        if (
            not math.isfinite(self.min_hand_confidence)
            or not 0.0 <= self.min_hand_confidence <= 1.0
        ):
            raise ValueError("min_hand_confidence must be in [0, 1]")

        object.__setattr__(self, "limits", MappingProxyType(validated_limits))
        object.__setattr__(self, "max_delta", MappingProxyType(validated_max_delta))


class TargetFilter:
    def __init__(self, config: SafetyConfig, initial_targets: RobotTargets) -> None:
        self.config = config
        self._validate_initial_targets(initial_targets)
        self._last = initial_targets

    @property
    def last_targets(self) -> RobotTargets:
        return self._last

    def update(
        self,
        desired: RobotTargets,
        *,
        now_ms: int,
        sample_timestamp_ms: int | None,
        sync_enabled: bool,
        neutral_ready: bool,
        deadman_active: bool,
        tracking_ok: bool,
    ) -> FilterResult:
        freeze_reason = self._freeze_reason(
            now_ms=now_ms,
            sample_timestamp_ms=sample_timestamp_ms,
            sync_enabled=sync_enabled,
            neutral_ready=neutral_ready,
            deadman_active=deadman_active,
            tracking_ok=tracking_ok,
        )
        if freeze_reason is not FreezeReason.ACTIVE:
            return FilterResult(
                self._last, frozen=True, clamped_keys=(), reason=freeze_reason
            )
        if not self._targets_are_finite(desired):
            return FilterResult(
                self._last,
                frozen=True,
                clamped_keys=(),
                reason=FreezeReason.TRACKING_LOST,
            )

        smoothed = RobotTargets(
            shoulder_pan=self._smooth(self._last.shoulder_pan, desired.shoulder_pan),
            shoulder_lift=self._smooth(self._last.shoulder_lift, desired.shoulder_lift),
            elbow_flex=self._smooth(self._last.elbow_flex, desired.elbow_flex),
            wrist_flex=self._smooth(self._last.wrist_flex, desired.wrist_flex),
            wrist_roll=self._smooth(self._last.wrist_roll, desired.wrist_roll),
            gripper=self._smooth(self._last.gripper, desired.gripper),
        )
        limited, clamped_keys = self._limit_delta_and_clamp(smoothed)
        self._last = limited
        return FilterResult(
            limited,
            frozen=False,
            clamped_keys=tuple(sorted(clamped_keys)),
            reason=FreezeReason.ACTIVE,
        )

    def _freeze_reason(
        self,
        *,
        now_ms: int,
        sample_timestamp_ms: int | None,
        sync_enabled: bool,
        neutral_ready: bool,
        deadman_active: bool,
        tracking_ok: bool,
    ) -> FreezeReason:
        if not sync_enabled:
            return FreezeReason.PAUSED
        if not neutral_ready:
            return FreezeReason.NEUTRAL_MISSING
        if not deadman_active:
            return FreezeReason.PAUSED
        if not tracking_ok or sample_timestamp_ms is None:
            return FreezeReason.TRACKING_LOST
        if now_ms - sample_timestamp_ms > self.config.stale_timeout_ms:
            return FreezeReason.STALE_RESULT
        return FreezeReason.ACTIVE

    def _smooth(self, previous: float, desired: float) -> float:
        alpha = self.config.smoothing
        return previous + alpha * (desired - previous)

    def _validate_initial_targets(self, initial_targets: RobotTargets) -> None:
        values = initial_targets.as_action()
        for key in CONTROLLED_KEYS:
            if not math.isfinite(values[key]):
                raise ValueError(f"initial target for {key} must be finite")
            low, high = self.config.limits[key]
            if not low <= values[key] <= high:
                raise ValueError(
                    f"initial target for {key} is outside safety limit "
                    f"[{low}, {high}]: {values[key]}"
                )

    def _targets_are_finite(self, targets: RobotTargets) -> bool:
        return all(math.isfinite(value) for value in targets.as_action().values())

    def _limit_delta_and_clamp(
        self, desired: RobotTargets
    ) -> tuple[RobotTargets, set[str]]:
        values = desired.as_action()
        previous = self._last.as_action()
        limited: dict[str, float] = {}
        clamped_keys: set[str] = set()

        for key in CONTROLLED_KEYS:
            delta = values[key] - previous[key]
            max_delta = self.config.max_delta[key]
            if delta > max_delta:
                values[key] = previous[key] + max_delta
                clamped_keys.add(key)
            elif delta < -max_delta:
                values[key] = previous[key] - max_delta
                clamped_keys.add(key)

            low, high = self.config.limits[key]
            if values[key] < low:
                values[key] = low
                clamped_keys.add(key)
            elif values[key] > high:
                values[key] = high
                clamped_keys.add(key)
            limited[key] = values[key]

        return (
            RobotTargets(
                shoulder_pan=limited["shoulder_pan.pos"],
                shoulder_lift=limited["shoulder_lift.pos"],
                elbow_flex=limited["elbow_flex.pos"],
                wrist_flex=limited["wrist_flex.pos"],
                wrist_roll=limited["wrist_roll.pos"],
                gripper=limited["gripper.pos"],
            ),
            clamped_keys,
        )
