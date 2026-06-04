# Egocentric SO101 6-DOF Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `playground/teleop/` — a single-RGB-webcam egocentric teleoperation prototype that controls all six SO101 follower joints from MediaPipe Pose + Hand Landmarker output, mirroring the four-stage tracker→mapper→safety→backend pipeline of `playground/mediapipe_so101/`.

**Architecture:** `pose_world_landmarks` (3D, hip-centered, meters) drives shoulder_pan/lift + elbow_flex through geometric joint-mimic math; the exocentric playground's 2D hand math drives wrist_flex/roll + gripper. Both perception models run concurrently in LIVE_STREAM mode with async result callbacks; a per-frame fusion step matches the closest Hand to the chosen Pose wrist. A 6-joint extension of the existing `TargetFilter` adds a visibility-based freeze rule for fail-safe behavior when arm landmarks leave frame.

**Tech Stack:** Python 3.12, uv, MediaPipe 0.10+ (Tasks API, both Pose and Hand Landmarker), OpenCV, LeRobot 0.5.1 (`SO101Follower`), pytest. Same dependency surface as `playground/mediapipe_so101/`.

---

## File Structure

```
playground/teleop/
├── README.md                       # Setup, dry-run/robot commands, controls, mount guidance, safety
├── pyproject.toml                  # uv project, same deps as mediapipe_so101
├── main.py                         # argparse, validate, camera loop, key handling, overlay
└── teleop/
    ├── __init__.py
    ├── types.py                    # Landmark, HandSample, PoseLandmark, ArmSample, TeleopSample,
    │                               # RobotTargets (6 joints), FilterResult, FreezeReason,
    │                               # ACTION_KEYS, CONTROLLED_KEYS
    ├── tracker.py                  # Pose + Hand model download/creation, camera open,
    │                               # LatestPoseResult, LatestHandResult, fusion (best_*_sample),
    │                               # frame_to_mp_image, draw_overlay
    ├── pose_mapper.py              # MappingConfig, ArmMapper, WristMapper, TeleopMapper
    ├── safety.py                   # SafetyConfig, TargetFilter (6 joints + visibility freeze)
    └── robot_backend.py            # Backend Protocol, DryRunBackend, SO101Backend (6 joints)
tests/
    ├── __init__.py
    ├── test_types.py               # RobotTargets.as_action() shape, FilterResult invariants
    ├── test_pose_mapper.py         # WristMapper + ArmMapper + TeleopMapper unit tests
    ├── test_safety.py              # 6-joint TargetFilter + visibility freeze
    ├── test_robot_backend.py       # DryRunBackend + SO101Backend with mocked RobotLike
    └── test_tracker.py             # Fusion logic + LatestResult thread safety + ensure_model
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `playground/teleop/pyproject.toml`
- Create: `playground/teleop/teleop/__init__.py`
- Create: `playground/teleop/tests/__init__.py`
- Create: `playground/teleop/tests/test_smoke.py`

- [ ] **Step 1: Create the pyproject.toml**

Create `playground/teleop/pyproject.toml` with the exact contents of `playground/mediapipe_so101/pyproject.toml`, only changing the name and description:

```toml
[project]
name = "teleop"
version = "0.1.0"
description = "Egocentric SO101 6-DOF teleoperation prototype using MediaPipe Pose + Hand."
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "feetech-servo-sdk>=1.0.0",
    "lerobot==0.5.1",
    "mediapipe>=0.10.35",
    "opencv-contrib-python>=4.13.0.92",
    "pytest>=8.4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create empty package init files**

Create `playground/teleop/teleop/__init__.py` with no content. Create `playground/teleop/tests/__init__.py` with no content.

- [ ] **Step 3: Create a smoke test that imports the package**

Create `playground/teleop/tests/test_smoke.py`:

```python
def test_package_imports() -> None:
    import teleop  # noqa: F401
```

- [ ] **Step 4: Sync deps and run the smoke test**

Run from `playground/teleop/`:

```bash
uv sync
uv run pytest tests/test_smoke.py -v
```

Expected: `test_package_imports PASSED`.

- [ ] **Step 5: Stub README and commit**

Create `playground/teleop/README.md`:

```markdown
# Egocentric SO101 Teleop

Egocentric webcam teleoperation prototype for the SO101 follower.
WIP — see docs/superpowers/specs/2026-05-24-egocentric-so101-teleop-design.md.
```

Commit:

```bash
git add playground/teleop/
git commit -m "feat(teleop): scaffold playground/teleop project"
```

---

## Task 2: Core types

**Files:**
- Create: `playground/teleop/teleop/types.py`
- Create: `playground/teleop/tests/test_types.py`

- [ ] **Step 1: Write failing tests for the types**

Create `playground/teleop/tests/test_types.py`:

```python
import math

import pytest

from teleop.types import (
    ACTION_KEYS,
    CONTROLLED_KEYS,
    ArmSample,
    FilterResult,
    FreezeReason,
    HandSample,
    Landmark,
    PoseLandmark,
    RobotTargets,
    TeleopSample,
)


def test_action_keys_cover_all_six_joints() -> None:
    assert ACTION_KEYS == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )


def test_controlled_keys_match_action_keys() -> None:
    assert CONTROLLED_KEYS == ACTION_KEYS


def test_robot_targets_as_action_returns_all_six_keys_in_order() -> None:
    targets = RobotTargets(
        shoulder_pan=1.0,
        shoulder_lift=2.0,
        elbow_flex=3.0,
        wrist_flex=4.0,
        wrist_roll=5.0,
        gripper=60.0,
    )

    action = targets.as_action()

    assert tuple(action.keys()) == ACTION_KEYS
    assert action["shoulder_pan.pos"] == 1.0
    assert action["shoulder_lift.pos"] == 2.0
    assert action["elbow_flex.pos"] == 3.0
    assert action["wrist_flex.pos"] == 4.0
    assert action["wrist_roll.pos"] == 5.0
    assert action["gripper.pos"] == 60.0


def test_freeze_reason_values_include_active_paused_neutral_tracking_stale() -> None:
    assert {reason.value for reason in FreezeReason} >= {
        "active",
        "paused",
        "neutral_missing",
        "tracking_lost",
        "stale_result",
    }


def test_filter_result_carries_targets_and_reason() -> None:
    targets = RobotTargets(0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    result = FilterResult(
        targets=targets,
        frozen=True,
        clamped_keys=("wrist_flex.pos",),
        reason=FreezeReason.PAUSED,
    )

    assert result.targets is targets
    assert result.frozen is True
    assert result.clamped_keys == ("wrist_flex.pos",)
    assert result.reason is FreezeReason.PAUSED


def test_landmark_defaults_z_to_zero() -> None:
    point = Landmark(x=0.1, y=0.2)
    assert point.z == 0.0


def test_pose_landmark_carries_visibility() -> None:
    point = PoseLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9)
    assert point.visibility == 0.9


def test_hand_sample_holds_21_landmarks_and_metadata() -> None:
    points = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    sample = HandSample(landmarks=points, handedness="Right", confidence=0.8, timestamp_ms=10)
    assert len(sample.landmarks) == 21
    assert sample.handedness == "Right"
    assert sample.confidence == 0.8
    assert sample.timestamp_ms == 10


def test_arm_sample_holds_shoulder_elbow_wrist_pose_landmarks_and_timestamp() -> None:
    shoulder = PoseLandmark(0.0, 0.0, 0.0, visibility=0.9)
    elbow = PoseLandmark(0.0, 0.3, 0.0, visibility=0.9)
    wrist = PoseLandmark(0.0, 0.6, 0.0, visibility=0.9)
    sample = ArmSample(
        shoulder=shoulder, elbow=elbow, wrist=wrist, wrist_image_xy=(0.5, 0.5), timestamp_ms=20
    )
    assert sample.shoulder is shoulder
    assert sample.elbow is elbow
    assert sample.wrist is wrist
    assert sample.wrist_image_xy == (0.5, 0.5)
    assert sample.timestamp_ms == 20


def test_teleop_sample_holds_arm_and_hand() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=20,
    )
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.8,
        timestamp_ms=20,
    )
    sample = TeleopSample(arm=arm, hand=hand, timestamp_ms=20)
    assert sample.arm is arm
    assert sample.hand is hand
    assert sample.timestamp_ms == 20


def test_teleop_sample_allows_missing_hand_or_arm() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=20,
    )
    sample = TeleopSample(arm=arm, hand=None, timestamp_ms=20)
    assert sample.arm is arm
    assert sample.hand is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shoulder_pan": math.nan},
        {"gripper": math.inf},
    ],
)
def test_robot_targets_accepts_non_finite_values(kwargs: dict) -> None:
    base = dict(
        shoulder_pan=0.0,
        shoulder_lift=0.0,
        elbow_flex=0.0,
        wrist_flex=0.0,
        wrist_roll=0.0,
        gripper=50.0,
    )
    base.update(kwargs)
    targets = RobotTargets(**base)
    assert isinstance(targets, RobotTargets)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_types.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name '...' from 'teleop.types'`.

- [ ] **Step 3: Implement types.py**

Create `playground/teleop/teleop/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

CONTROLLED_KEYS = ACTION_KEYS


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class HandSample:
    landmarks: Sequence[Landmark]
    handedness: str
    confidence: float
    timestamp_ms: int


@dataclass(frozen=True)
class ArmSample:
    shoulder: PoseLandmark
    elbow: PoseLandmark
    wrist: PoseLandmark
    wrist_image_xy: tuple[float, float]
    timestamp_ms: int


@dataclass(frozen=True)
class TeleopSample:
    arm: ArmSample | None
    hand: HandSample | None
    timestamp_ms: int


@dataclass(frozen=True)
class RobotTargets:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float

    def as_action(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": self.shoulder_pan,
            "shoulder_lift.pos": self.shoulder_lift,
            "elbow_flex.pos": self.elbow_flex,
            "wrist_flex.pos": self.wrist_flex,
            "wrist_roll.pos": self.wrist_roll,
            "gripper.pos": self.gripper,
        }


class FreezeReason(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    NEUTRAL_MISSING = "neutral_missing"
    TRACKING_LOST = "tracking_lost"
    STALE_RESULT = "stale_result"


@dataclass(frozen=True)
class FilterResult:
    targets: RobotTargets
    frozen: bool
    clamped_keys: tuple[str, ...]
    reason: FreezeReason
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_types.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/types.py playground/teleop/tests/test_types.py
git commit -m "feat(teleop): add core types for egocentric 6-DOF teleop"
```

---

## Task 3: WristMapper (ported from exocentric)

**Files:**
- Create: `playground/teleop/teleop/pose_mapper.py` (partial — MappingConfig, HandFeatures, WristMapper, extract_wrist_features)
- Create: `playground/teleop/tests/test_pose_mapper.py` (partial — wrist tests only for this task)

- [ ] **Step 1: Write failing wrist tests**

Create `playground/teleop/tests/test_pose_mapper.py`:

```python
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
    config = MappingConfig(
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
        **overrides,
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name '...' from 'teleop.pose_mapper'`.

- [ ] **Step 3: Implement MappingConfig, HandFeatures, WristMapper**

Create `playground/teleop/teleop/pose_mapper.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .types import HandSample, Landmark, RobotTargets


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17


@dataclass(frozen=True)
class MappingConfig:
    shoulder_pan_gain: float = 30.0
    shoulder_lift_gain: float = 30.0
    elbow_flex_gain: float = 30.0
    wrist_flex_gain: float = 30.0
    wrist_roll_gain: float = 60.0
    gripper_open: float = 80.0
    gripper_closed: float = 20.0
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.40
    min_hand_width: float = 0.03
    min_arm_segment: float = 0.05
    mirror_hand: bool = False

    def __post_init__(self) -> None:
        numeric_fields = (
            "shoulder_pan_gain",
            "shoulder_lift_gain",
            "elbow_flex_gain",
            "wrist_flex_gain",
            "wrist_roll_gain",
            "gripper_open",
            "gripper_closed",
            "pinch_closed_ratio",
            "pinch_open_ratio",
            "min_hand_width",
            "min_arm_segment",
        )
        for field_name in numeric_fields:
            _validate_finite_number(field_name, getattr(self, field_name))

        if self.min_hand_width <= 0.0:
            raise ValueError("min_hand_width must be greater than 0")
        if self.min_arm_segment <= 0.0:
            raise ValueError("min_arm_segment must be greater than 0")
        if self.pinch_open_ratio <= self.pinch_closed_ratio:
            raise ValueError("pinch_open_ratio must be greater than pinch_closed_ratio")


@dataclass(frozen=True)
class HandFeatures:
    roll: float
    flex: float
    pinch_open: float


class WristMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: HandFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: HandSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_wrist_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: HandSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral wrist features have not been captured")

        features = extract_wrist_features(sample, self.config)
        mirror = -1.0 if self.config.mirror_hand else 1.0
        flex_delta = features.flex - self._neutral_features.flex
        roll_delta = mirror * _normalize_angle(features.roll - self._neutral_features.roll)
        gripper = self.config.gripper_closed + features.pinch_open * (
            self.config.gripper_open - self.config.gripper_closed
        )

        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan,
            shoulder_lift=self._neutral_targets.shoulder_lift,
            elbow_flex=self._neutral_targets.elbow_flex,
            wrist_flex=self._neutral_targets.wrist_flex + flex_delta * self.config.wrist_flex_gain,
            wrist_roll=self._neutral_targets.wrist_roll + roll_delta * self.config.wrist_roll_gain,
            gripper=gripper,
        )


def extract_wrist_features(sample: HandSample, config: MappingConfig | None = None) -> HandFeatures:
    cfg = config or MappingConfig()
    landmarks = sample.landmarks
    if len(landmarks) < 21:
        raise ValueError("HandSample must contain 21 landmarks")

    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]

    for landmark in (wrist, thumb_tip, index_mcp, index_tip, middle_mcp, pinky_mcp):
        _validate_landmark(landmark)

    hand_width = _distance3(index_mcp, pinky_mcp)
    if hand_width < cfg.min_hand_width:
        raise ValueError("Hand width is too small")

    roll = math.atan2(index_mcp.y - pinky_mcp.y, pinky_mcp.x - index_mcp.x)
    flex = (middle_mcp.y - wrist.y) / hand_width
    pinch_ratio = _distance3(thumb_tip, index_tip) / hand_width
    pinch_open = _clamp(
        (pinch_ratio - cfg.pinch_closed_ratio) / (cfg.pinch_open_ratio - cfg.pinch_closed_ratio),
        0.0,
        1.0,
    )

    return HandFeatures(roll=roll, flex=flex, pinch_open=pinch_open)


def _distance3(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _validate_landmark(landmark: Landmark) -> None:
    if not all(math.isfinite(value) for value in (landmark.x, landmark.y, landmark.z)):
        raise ValueError("Landmark coordinates must be finite")


def _validate_robot_targets(robot_targets: RobotTargets) -> None:
    values = (
        robot_targets.shoulder_pan,
        robot_targets.shoulder_lift,
        robot_targets.elbow_flex,
        robot_targets.wrist_flex,
        robot_targets.wrist_roll,
        robot_targets.gripper,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Robot target values must be finite")


def _validate_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All wrist tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/pose_mapper.py playground/teleop/tests/test_pose_mapper.py
git commit -m "feat(teleop): add WristMapper ported from exocentric playground"
```

---

## Task 4: ArmMapper (new geometric math)

**Files:**
- Modify: `playground/teleop/teleop/pose_mapper.py` (add `ArmMapper`, `ArmFeatures`, `extract_arm_features`)
- Modify: `playground/teleop/tests/test_pose_mapper.py` (append arm tests)

- [ ] **Step 1: Write failing arm tests**

Append to `playground/teleop/tests/test_pose_mapper.py`:

```python
from teleop.pose_mapper import ArmMapper, extract_arm_features
from teleop.types import ArmSample, PoseLandmark


def arm(
    *,
    shoulder=(0.0, 0.0, 0.0),
    elbow=(0.0, 0.3, 0.0),
    wrist=(0.0, 0.6, 0.0),
    visibility=0.9,
    wrist_image_xy=(0.5, 0.5),
    timestamp_ms=1000,
) -> ArmSample:
    return ArmSample(
        shoulder=PoseLandmark(*shoulder, visibility=visibility),
        elbow=PoseLandmark(*elbow, visibility=visibility),
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: New arm tests FAIL with `ImportError: cannot import name 'ArmMapper'`.

- [ ] **Step 3: Add ArmFeatures, extract_arm_features, ArmMapper to pose_mapper.py**

Append to `playground/teleop/teleop/pose_mapper.py`:

```python
from .types import ArmSample, PoseLandmark


@dataclass(frozen=True)
class ArmFeatures:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float


def extract_arm_features(sample: ArmSample, config: MappingConfig | None = None) -> ArmFeatures:
    cfg = config or MappingConfig()
    for landmark in (sample.shoulder, sample.elbow, sample.wrist):
        _validate_pose_landmark(landmark)

    upper_arm = (
        sample.elbow.x - sample.shoulder.x,
        sample.elbow.y - sample.shoulder.y,
        sample.elbow.z - sample.shoulder.z,
    )
    forearm = (
        sample.wrist.x - sample.elbow.x,
        sample.wrist.y - sample.elbow.y,
        sample.wrist.z - sample.elbow.z,
    )

    upper_arm_length = math.sqrt(sum(component * component for component in upper_arm))
    forearm_length = math.sqrt(sum(component * component for component in forearm))
    if upper_arm_length < cfg.min_arm_segment:
        raise ValueError("Upper arm segment is too short")
    if forearm_length < cfg.min_arm_segment:
        raise ValueError("Forearm segment is too short")

    cos_elbow = sum(u * f for u, f in zip(upper_arm, forearm)) / (upper_arm_length * forearm_length)
    cos_elbow = _clamp(cos_elbow, -1.0, 1.0)
    elbow_flex = math.acos(cos_elbow)

    horizontal = math.sqrt(upper_arm[0] ** 2 + upper_arm[2] ** 2)
    shoulder_lift = math.atan2(horizontal, upper_arm[1])
    shoulder_pan = math.atan2(upper_arm[0], upper_arm[2])

    return ArmFeatures(
        shoulder_pan=shoulder_pan,
        shoulder_lift=shoulder_lift,
        elbow_flex=elbow_flex,
    )


def _validate_pose_landmark(landmark: PoseLandmark) -> None:
    if not all(
        math.isfinite(value) for value in (landmark.x, landmark.y, landmark.z, landmark.visibility)
    ):
        raise ValueError("Pose landmark coordinates and visibility must be finite")


class ArmMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: ArmFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: ArmSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_arm_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: ArmSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral arm features have not been captured")

        features = extract_arm_features(sample, self.config)
        pan_delta = _normalize_angle(features.shoulder_pan - self._neutral_features.shoulder_pan)
        lift_delta = features.shoulder_lift - self._neutral_features.shoulder_lift
        elbow_delta = features.elbow_flex - self._neutral_features.elbow_flex

        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan
            + pan_delta * self.config.shoulder_pan_gain,
            shoulder_lift=self._neutral_targets.shoulder_lift
            + lift_delta * self.config.shoulder_lift_gain,
            elbow_flex=self._neutral_targets.elbow_flex
            + elbow_delta * self.config.elbow_flex_gain,
            wrist_flex=self._neutral_targets.wrist_flex,
            wrist_roll=self._neutral_targets.wrist_roll,
            gripper=self._neutral_targets.gripper,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All pose_mapper tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/pose_mapper.py playground/teleop/tests/test_pose_mapper.py
git commit -m "feat(teleop): add ArmMapper with geometric joint-mimic math"
```

---

## Task 5: TeleopMapper (combine arm + wrist)

**Files:**
- Modify: `playground/teleop/teleop/pose_mapper.py` (add `TeleopMapper`)
- Modify: `playground/teleop/tests/test_pose_mapper.py` (append combined tests)

- [ ] **Step 1: Write failing combined mapper tests**

Append to `playground/teleop/tests/test_pose_mapper.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: New TeleopMapper tests FAIL with `ImportError`.

- [ ] **Step 3: Add TeleopMapper to pose_mapper.py**

Append to `playground/teleop/teleop/pose_mapper.py`:

```python
from .types import TeleopSample


class TeleopMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.arm = ArmMapper(config)
        self.wrist = WristMapper(config)

    @property
    def neutral_ready(self) -> bool:
        return self.arm.neutral_ready and self.wrist.neutral_ready

    def capture_neutral(self, sample: TeleopSample, robot_targets: RobotTargets) -> None:
        if sample.arm is None or sample.hand is None:
            raise ValueError("TeleopMapper neutral capture requires both arm and hand samples")
        self.arm.capture_neutral(sample.arm, robot_targets)
        self.wrist.capture_neutral(sample.hand, robot_targets)

    def map(self, sample: TeleopSample) -> RobotTargets:
        if sample.arm is None or sample.hand is None:
            raise ValueError("TeleopMapper map requires both arm and hand samples")
        if not self.neutral_ready:
            raise RuntimeError("Neutral TeleopMapper sample has not been captured")

        arm_targets = self.arm.map(sample.arm)
        wrist_targets = self.wrist.map(sample.hand)
        return RobotTargets(
            shoulder_pan=arm_targets.shoulder_pan,
            shoulder_lift=arm_targets.shoulder_lift,
            elbow_flex=arm_targets.elbow_flex,
            wrist_flex=wrist_targets.wrist_flex,
            wrist_roll=wrist_targets.wrist_roll,
            gripper=wrist_targets.gripper,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/pose_mapper.py playground/teleop/tests/test_pose_mapper.py
git commit -m "feat(teleop): add TeleopMapper combining arm and wrist halves"
```

---

## Task 6: SafetyConfig and TargetFilter for 6 joints

**Files:**
- Create: `playground/teleop/teleop/safety.py`
- Create: `playground/teleop/tests/test_safety.py`

- [ ] **Step 1: Write failing safety tests**

Create `playground/teleop/tests/test_safety.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_safety.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name 'SafetyConfig'`.

- [ ] **Step 3: Implement safety.py**

Create `playground/teleop/teleop/safety.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_safety.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/safety.py playground/teleop/tests/test_safety.py
git commit -m "feat(teleop): add 6-joint TargetFilter with visibility/confidence config"
```

---

## Task 7: DryRunBackend and SO101Backend (6 joints)

**Files:**
- Create: `playground/teleop/teleop/robot_backend.py`
- Create: `playground/teleop/tests/test_robot_backend.py`

- [ ] **Step 1: Write failing backend tests**

Create `playground/teleop/tests/test_robot_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_robot_backend.py -v
```

Expected: All tests FAIL with `ImportError`.

- [ ] **Step 3: Implement robot_backend.py**

Create `playground/teleop/teleop/robot_backend.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_robot_backend.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/robot_backend.py playground/teleop/tests/test_robot_backend.py
git commit -m "feat(teleop): add 6-DOF DryRunBackend and SO101Backend"
```

---

## Task 8: Tracker — model paths, ensure_model, camera helpers

**Files:**
- Create: `playground/teleop/teleop/tracker.py` (model helpers + camera + frame_to_mp_image only — landmarker creation and fusion in Task 9)
- Create: `playground/teleop/tests/test_tracker.py` (model path tests for this task)

- [ ] **Step 1: Write failing tracker tests for model paths and ensure_model**

Create `playground/teleop/tests/test_tracker.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from teleop.tracker import (
    default_hand_model_path,
    default_pose_model_path,
    ensure_model,
)


def test_default_pose_model_path_lives_under_models_dir() -> None:
    path = default_pose_model_path()
    assert path.name == "pose_landmarker_lite.task"
    assert path.parent.name == "models"


def test_default_hand_model_path_lives_under_models_dir() -> None:
    path = default_hand_model_path()
    assert path.name == "hand_landmarker.task"
    assert path.parent.name == "models"


def test_ensure_model_returns_existing_path_when_file_present(tmp_path: Path) -> None:
    file_path = tmp_path / "existing.task"
    file_path.write_bytes(b"\x00")
    assert ensure_model(file_path, url="https://example.invalid/model.task") == file_path.resolve()


def test_ensure_model_downloads_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "missing.task"

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self._sent = False

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return self._payload

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        assert url == "https://example.invalid/model.task"
        return FakeResponse(b"abc")

    with patch("teleop.tracker.urllib.request.urlopen", side_effect=fake_urlopen):
        result = ensure_model(target, url="https://example.invalid/model.task")
    assert result == target.resolve()
    assert target.read_bytes() == b"abc"


def test_ensure_model_cleans_up_temp_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "fail.task"

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    with patch("teleop.tracker.urllib.request.urlopen", side_effect=boom):
        with pytest.raises(RuntimeError, match="network down"):
            ensure_model(target, url="https://example.invalid/model.task")
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: All tests FAIL with `ImportError`.

- [ ] **Step 3: Implement tracker.py with model helpers + camera + frame conversion**

Create `playground/teleop/teleop/tracker.py`:

```python
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def default_pose_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"


def default_hand_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def ensure_model(model_path: Path, *, url: str) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"Downloading MediaPipe model to {model_path}")
    tmp_path.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with tmp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp_path.replace(model_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return model_path


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    default_backend = getattr(cv2, "CAP_ANY", 0)
    api_preference = (
        getattr(cv2, "CAP_AVFOUNDATION", default_backend)
        if sys.platform == "darwin"
        else default_backend
    )
    capture = cv2.VideoCapture(camera_index, api_preference)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "On macOS, make sure the terminal app has camera permission."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        capture.read()
    return capture


def frame_to_mp_image(frame) -> mp.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/tracker.py playground/teleop/tests/test_tracker.py
git commit -m "feat(teleop): add Pose+Hand model helpers and camera open"
```

---

## Task 9: Tracker — landmarker creation, latest-result buffers, fusion

**Files:**
- Modify: `playground/teleop/teleop/tracker.py` (add landmarker creators, `LatestPoseResult`, `LatestHandResult`, `fuse_samples`)
- Modify: `playground/teleop/tests/test_tracker.py` (append fusion tests)

- [ ] **Step 1: Write failing fusion tests**

Append to `playground/teleop/tests/test_tracker.py`:

```python
import threading

from teleop.tracker import (
    LatestHandResult,
    LatestPoseResult,
    fuse_samples,
)
from teleop.types import ArmSample, HandSample, Landmark, PoseLandmark


def make_pose_landmark_obj(x: float, y: float, z: float, visibility: float):
    class _L:
        pass

    obj = _L()
    obj.x = x
    obj.y = y
    obj.z = z
    obj.visibility = visibility
    obj.presence = visibility
    return obj


def make_pose_result(
    *,
    image_landmarks_per_arm: dict[str, tuple[float, float, float, float]] | None = None,
    world_landmarks_per_arm: dict[str, tuple[float, float, float, float]] | None = None,
) -> object:
    class _Result:
        pass

    result = _Result()
    image_defaults = {
        "shoulder": (0.5, 0.3, 0.0, 0.95),
        "elbow": (0.6, 0.5, 0.0, 0.95),
        "wrist": (0.7, 0.7, 0.0, 0.95),
    }
    world_defaults = {
        "shoulder": (0.0, 0.0, 0.0, 0.95),
        "elbow": (0.1, 0.2, 0.0, 0.95),
        "wrist": (0.15, 0.4, 0.0, 0.95),
    }
    image_data = image_landmarks_per_arm or image_defaults
    world_data = world_landmarks_per_arm or world_defaults

    image_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    world_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    image_landmarks[12] = make_pose_landmark_obj(*image_data["shoulder"])
    image_landmarks[14] = make_pose_landmark_obj(*image_data["elbow"])
    image_landmarks[16] = make_pose_landmark_obj(*image_data["wrist"])
    world_landmarks[12] = make_pose_landmark_obj(*world_data["shoulder"])
    world_landmarks[14] = make_pose_landmark_obj(*world_data["elbow"])
    world_landmarks[16] = make_pose_landmark_obj(*world_data["wrist"])

    result.pose_landmarks = [image_landmarks]
    result.pose_world_landmarks = [world_landmarks]
    return result


def make_hand_result(hands: list[dict]) -> object:
    class _Result:
        pass

    result = _Result()
    hand_landmarks_lists = []
    handedness_lists = []
    for hand in hands:
        points = [make_pose_landmark_obj(hand["x"], hand["y"], 0.0, 0.0) for _ in range(21)]
        hand_landmarks_lists.append(points)
        handedness_objs = []

        class _H:
            pass

        h = _H()
        h.category_name = hand.get("handedness", "Right")
        h.score = hand.get("score", 0.9)
        handedness_objs.append(h)
        handedness_lists.append(handedness_objs)
    result.hand_landmarks = hand_landmarks_lists
    result.handedness = handedness_lists
    return result


def test_latest_pose_result_returns_none_when_no_callback_received() -> None:
    latest = LatestPoseResult()
    assert latest.best_arm_sample(arm="right") is None


def test_latest_pose_result_returns_arm_sample_for_chosen_arm() -> None:
    latest = LatestPoseResult()
    latest.update(make_pose_result(), None, timestamp_ms=10)

    sample = latest.best_arm_sample(arm="right")
    assert sample is not None
    assert sample.timestamp_ms == 10
    assert sample.shoulder.visibility == pytest.approx(0.95)
    assert sample.wrist_image_xy == pytest.approx((0.7, 0.7))


def test_latest_pose_result_supports_left_arm() -> None:
    latest = LatestPoseResult()
    image_data = {
        "shoulder": (0.4, 0.3, 0.0, 0.8),
        "elbow": (0.3, 0.5, 0.0, 0.8),
        "wrist": (0.2, 0.7, 0.0, 0.8),
    }
    world_data = {
        "shoulder": (0.0, 0.0, 0.0, 0.8),
        "elbow": (-0.1, 0.2, 0.0, 0.8),
        "wrist": (-0.15, 0.4, 0.0, 0.8),
    }
    # Left side indices: shoulder=11, elbow=13, wrist=15
    class _Result:
        pass

    result = _Result()
    image_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    world_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    image_landmarks[11] = make_pose_landmark_obj(*image_data["shoulder"])
    image_landmarks[13] = make_pose_landmark_obj(*image_data["elbow"])
    image_landmarks[15] = make_pose_landmark_obj(*image_data["wrist"])
    world_landmarks[11] = make_pose_landmark_obj(*world_data["shoulder"])
    world_landmarks[13] = make_pose_landmark_obj(*world_data["elbow"])
    world_landmarks[15] = make_pose_landmark_obj(*world_data["wrist"])
    result.pose_landmarks = [image_landmarks]
    result.pose_world_landmarks = [world_landmarks]

    latest.update(result, None, timestamp_ms=20)
    sample = latest.best_arm_sample(arm="left")
    assert sample is not None
    assert sample.wrist_image_xy == pytest.approx((0.2, 0.7))


def test_latest_hand_result_returns_none_when_no_callback_received() -> None:
    latest = LatestHandResult()
    assert latest.best_hand_sample() is None


def test_latest_hand_result_returns_highest_confidence_sample() -> None:
    latest = LatestHandResult()
    result = make_hand_result(
        [
            {"x": 0.4, "y": 0.5, "score": 0.5},
            {"x": 0.7, "y": 0.7, "score": 0.95},
        ]
    )
    latest.update(result, None, timestamp_ms=30)
    sample = latest.best_hand_sample()
    assert sample is not None
    assert sample.confidence == pytest.approx(0.95)
    assert sample.landmarks[0].x == pytest.approx(0.7)


def test_latest_result_buffers_are_thread_safe() -> None:
    latest = LatestHandResult()
    result_a = make_hand_result([{"x": 0.5, "y": 0.5, "score": 0.7}])
    result_b = make_hand_result([{"x": 0.5, "y": 0.5, "score": 0.7}])

    def writer(result, timestamp_ms):
        for _ in range(100):
            latest.update(result, None, timestamp_ms=timestamp_ms)

    threads = [
        threading.Thread(target=writer, args=(result_a, 10)),
        threading.Thread(target=writer, args=(result_b, 20)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sample = latest.best_hand_sample()
    assert sample is not None
    assert sample.timestamp_ms in (10, 20)


def test_fuse_samples_pairs_arm_with_nearest_hand_in_image_plane() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.75, 0.75),
        timestamp_ms=10,
    )
    far_hand = HandSample(
        landmarks=[Landmark(0.10, 0.10, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    near_hand = HandSample(
        landmarks=[Landmark(0.74, 0.76, 0.0) for _ in range(21)],
        handedness="Left",
        confidence=0.9,
        timestamp_ms=10,
    )

    sample = fuse_samples(arm=arm, hands=[far_hand, near_hand], timestamp_ms=10)
    assert sample.arm is arm
    assert sample.hand is near_hand


def test_fuse_samples_returns_none_hand_when_no_hands_present() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=10,
    )
    sample = fuse_samples(arm=arm, hands=[], timestamp_ms=10)
    assert sample.arm is arm
    assert sample.hand is None


def test_fuse_samples_returns_none_arm_when_arm_missing() -> None:
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    sample = fuse_samples(arm=None, hands=[hand], timestamp_ms=10)
    assert sample.arm is None
    assert sample.hand is hand
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: New tests FAIL with `ImportError`.

- [ ] **Step 3: Add landmarker creators, latest-result buffers, and fusion to tracker.py**

Append to `playground/teleop/teleop/tracker.py`:

```python
import threading

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .types import ArmSample, HandSample, Landmark, PoseLandmark, TeleopSample


POSE_RIGHT_SHOULDER = 12
POSE_RIGHT_ELBOW = 14
POSE_RIGHT_WRIST = 16
POSE_LEFT_SHOULDER = 11
POSE_LEFT_ELBOW = 13
POSE_LEFT_WRIST = 15


def create_pose_landmarker(
    *,
    model_path: Path,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    result_callback,
) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        output_segmentation_masks=False,
        result_callback=result_callback,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_landmarker(
    *,
    model_path: Path,
    max_hands: int,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    result_callback,
) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        result_callback=result_callback,
    )
    return vision.HandLandmarker.create_from_options(options)


class LatestPoseResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None
        self._timestamp_ms = -1

    def update(self, result, _output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def best_arm_sample(self, arm: str) -> ArmSample | None:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        if result is None or not result.pose_landmarks or not result.pose_world_landmarks:
            return None

        if arm == "right":
            indices = (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST)
        elif arm == "left":
            indices = (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST)
        else:
            raise ValueError(f"Unknown arm selection: {arm}")

        image_lms = result.pose_landmarks[0]
        world_lms = result.pose_world_landmarks[0]
        shoulder_world = world_lms[indices[0]]
        elbow_world = world_lms[indices[1]]
        wrist_world = world_lms[indices[2]]
        wrist_image = image_lms[indices[2]]

        return ArmSample(
            shoulder=PoseLandmark(
                shoulder_world.x,
                shoulder_world.y,
                shoulder_world.z,
                visibility=shoulder_world.visibility,
            ),
            elbow=PoseLandmark(
                elbow_world.x, elbow_world.y, elbow_world.z, visibility=elbow_world.visibility
            ),
            wrist=PoseLandmark(
                wrist_world.x, wrist_world.y, wrist_world.z, visibility=wrist_world.visibility
            ),
            wrist_image_xy=(wrist_image.x, wrist_image.y),
            timestamp_ms=timestamp_ms,
        )


class LatestHandResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None
        self._timestamp_ms = -1

    def update(self, result, _output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def best_hand_sample(self) -> HandSample | None:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        return _extract_best_hand(result, timestamp_ms)

    def all_hand_samples(self) -> list[HandSample]:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        if result is None or not result.hand_landmarks:
            return []
        samples: list[HandSample] = []
        for idx in range(len(result.hand_landmarks)):
            handedness = result.handedness[idx] if idx < len(result.handedness) else []
            label = handedness[0].category_name if handedness else "Hand"
            score = handedness[0].score if handedness else 0.0
            landmarks = [
                Landmark(x=point.x, y=point.y, z=point.z)
                for point in result.hand_landmarks[idx]
            ]
            samples.append(
                HandSample(
                    landmarks=landmarks,
                    handedness=label,
                    confidence=score,
                    timestamp_ms=timestamp_ms,
                )
            )
        return samples


def _extract_best_hand(result, timestamp_ms: int) -> HandSample | None:
    if result is None or not result.hand_landmarks:
        return None
    best_index = 0
    best_score = -1.0
    for idx in range(len(result.hand_landmarks)):
        handedness = result.handedness[idx] if idx < len(result.handedness) else []
        score = handedness[0].score if handedness else 0.0
        if score > best_score:
            best_index = idx
            best_score = score
    landmarks = [
        Landmark(x=point.x, y=point.y, z=point.z)
        for point in result.hand_landmarks[best_index]
    ]
    handedness = result.handedness[best_index] if best_index < len(result.handedness) else []
    label = handedness[0].category_name if handedness else "Hand"
    score = handedness[0].score if handedness else 0.0
    return HandSample(landmarks=landmarks, handedness=label, confidence=score, timestamp_ms=timestamp_ms)


def fuse_samples(
    *, arm: ArmSample | None, hands: list[HandSample], timestamp_ms: int
) -> TeleopSample:
    if arm is None:
        chosen_hand = max(hands, key=lambda h: h.confidence, default=None)
        return TeleopSample(arm=None, hand=chosen_hand, timestamp_ms=timestamp_ms)

    if not hands:
        return TeleopSample(arm=arm, hand=None, timestamp_ms=timestamp_ms)

    wrist_x, wrist_y = arm.wrist_image_xy

    def distance(hand: HandSample) -> float:
        wrist_landmark = hand.landmarks[0]
        return (wrist_landmark.x - wrist_x) ** 2 + (wrist_landmark.y - wrist_y) ** 2

    chosen_hand = min(hands, key=distance)
    return TeleopSample(arm=arm, hand=chosen_hand, timestamp_ms=timestamp_ms)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/tracker.py playground/teleop/tests/test_tracker.py
git commit -m "feat(teleop): add Pose+Hand landmarker creation, latest buffers, fusion"
```

---

## Task 10: Tracker — overlay drawing

**Files:**
- Modify: `playground/teleop/teleop/tracker.py` (add `draw_overlay`)
- Modify: `playground/teleop/tests/test_tracker.py` (append overlay smoke tests)

- [ ] **Step 1: Write failing overlay tests**

Append to `playground/teleop/tests/test_tracker.py`:

```python
import numpy as np

from teleop.tracker import draw_overlay


def test_draw_overlay_does_not_modify_when_samples_are_none() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    snapshot = frame.copy()
    draw_overlay(
        frame,
        arm=None,
        hand=None,
        status_lines=[],
        image_size=(320, 240),
    )
    assert np.array_equal(frame, snapshot)


def test_draw_overlay_draws_pose_skeleton_lines_when_arm_present() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=10,
    )
    image_landmarks = {
        "shoulder": (0.2, 0.2),
        "elbow": (0.4, 0.4),
        "wrist": (0.5, 0.5),
    }
    draw_overlay(
        frame,
        arm=arm,
        hand=None,
        status_lines=["status text"],
        image_size=(320, 240),
        arm_image_landmarks=image_landmarks,
    )
    assert frame.sum() > 0


def test_draw_overlay_draws_hand_landmarks_when_hand_present() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    draw_overlay(
        frame,
        arm=None,
        hand=hand,
        status_lines=[],
        image_size=(320, 240),
    )
    assert frame.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: New overlay tests FAIL with `ImportError: cannot import name 'draw_overlay'`.

- [ ] **Step 3: Implement draw_overlay**

Append to `playground/teleop/teleop/tracker.py`:

```python
HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


def draw_overlay(
    frame,
    *,
    arm: ArmSample | None,
    hand: HandSample | None,
    status_lines: list[str],
    image_size: tuple[int, int],
    arm_image_landmarks: dict[str, tuple[float, float]] | None = None,
) -> None:
    width, height = image_size

    if arm is not None and arm_image_landmarks is not None:
        labels = ("shoulder", "elbow", "wrist")
        visibilities = (arm.shoulder.visibility, arm.elbow.visibility, arm.wrist.visibility)
        points = []
        for label in labels:
            normalized = arm_image_landmarks.get(label)
            if normalized is None:
                points.append(None)
                continue
            px = max(0, min(width - 1, int(normalized[0] * width)))
            py = max(0, min(height - 1, int(normalized[1] * height)))
            points.append((px, py))
        for start, end in ((0, 1), (1, 2)):
            if points[start] is None or points[end] is None:
                continue
            cv2.line(frame, points[start], points[end], (255, 180, 70), 3, cv2.LINE_AA)
        for label, point, visibility in zip(labels, points, visibilities):
            if point is None:
                continue
            cv2.circle(frame, point, 7, (255, 220, 130), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{visibility:.2f}",
                (point[0] + 8, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 130),
                1,
                cv2.LINE_AA,
            )

    if hand is not None:
        hand_points = [
            (
                max(0, min(width - 1, int(point.x * width))),
                max(0, min(height - 1, int(point.y * height))),
            )
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            if start >= len(hand_points) or end >= len(hand_points):
                continue
            cv2.line(frame, hand_points[start], hand_points[end], (65, 210, 120), 2, cv2.LINE_AA)
        for index, point in enumerate(hand_points):
            radius = 6 if index in {4, 8, 12, 16, 20} else 4
            cv2.circle(frame, point, radius, (35, 115, 255), -1, cv2.LINE_AA)

    for index, line in enumerate(status_lines):
        y = 30 + index * 22
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_tracker.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/teleop/tracker.py playground/teleop/tests/test_tracker.py
git commit -m "feat(teleop): add draw_overlay for pose, hand, and status text"
```

---

## Task 11: main.py — argparse and validation

**Files:**
- Create: `playground/teleop/main.py` (argparse + validation + camera defaults; loop deferred to Task 12)
- Create: `playground/teleop/tests/test_main_args.py`

- [ ] **Step 1: Write failing argument tests**

Create `playground/teleop/tests/test_main_args.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_main_args.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Implement main.py with parse_args, apply_camera_defaults, validate_args**

Create `playground/teleop/main.py`:

```python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from teleop.pose_mapper import MappingConfig
from teleop.tracker import default_hand_model_path, default_pose_model_path
from teleop.types import RobotTargets


WINDOW_NAME = "Egocentric SO101 Teleop"

CAMERA_DEFAULTS_BY_INDEX = {
    0: (640, 480, 30),
}
FALLBACK_CAMERA_DEFAULTS = (1280, 720, 30)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Egocentric SO101 6-DOF teleoperation from MediaPipe Pose + Hand."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--check", action="store_true")

    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--mirror-hand", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--presence-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-confidence", type=float, default=0.45)
    parser.add_argument("--min-pose-visibility", type=float, default=0.6)

    parser.add_argument("--pose-model-path", type=Path, default=default_pose_model_path())
    parser.add_argument("--hand-model-path", type=Path, default=default_hand_model_path())

    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--robot-port", type=str)
    parser.add_argument("--robot-id", type=str)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("../so101/calibration/robots/so_follower"),
    )
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--deadman-key", type=str, default="")
    parser.add_argument("--deadman-grace-ms", type=int, default=175)

    parser.add_argument("--shoulder-pan-gain", type=float, default=20.0)
    parser.add_argument("--shoulder-lift-gain", type=float, default=20.0)
    parser.add_argument("--elbow-flex-gain", type=float, default=20.0)
    parser.add_argument("--wrist-flex-gain", type=float, default=30.0)
    parser.add_argument("--wrist-roll-gain", type=float, default=60.0)
    parser.add_argument("--gripper-open", type=float, default=80.0)
    parser.add_argument("--gripper-closed", type=float, default=20.0)
    parser.add_argument("--pinch-closed-ratio", type=float, default=0.35)
    parser.add_argument("--pinch-open-ratio", type=float, default=1.40)

    parser.add_argument("--shoulder-pan-limit", type=float, default=20.0)
    parser.add_argument("--shoulder-lift-limit", type=float, default=20.0)
    parser.add_argument("--elbow-flex-limit", type=float, default=25.0)
    parser.add_argument("--wrist-flex-limit", type=float, default=15.0)
    parser.add_argument("--wrist-roll-limit", type=float, default=25.0)
    parser.add_argument("--gripper-min", type=float, default=15.0)
    parser.add_argument("--gripper-max", type=float, default=85.0)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--stale-timeout-ms", type=int, default=200)

    return parser.parse_args(argv)


def apply_camera_defaults(args: argparse.Namespace) -> None:
    width, height, fps = CAMERA_DEFAULTS_BY_INDEX.get(args.camera_index, FALLBACK_CAMERA_DEFAULTS)
    if args.width is None:
        args.width = width
    if args.height is None:
        args.height = height
    if args.fps is None:
        args.fps = fps


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.max_hands <= 0:
        raise SystemExit("--max-hands must be positive")
    if args.deadman_key and len(args.deadman_key) != 1:
        raise SystemExit("--deadman-key must be a single character")
    if args.deadman_grace_ms < 0:
        raise SystemExit("--deadman-grace-ms must be non-negative")

    for name in (
        "detection_confidence",
        "presence_confidence",
        "tracking_confidence",
        "min_hand_confidence",
        "min_pose_visibility",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.gripper_min >= args.gripper_max:
        raise SystemExit("--gripper-min must be less than --gripper-max")

    try:
        MappingConfig(
            shoulder_pan_gain=args.shoulder_pan_gain,
            shoulder_lift_gain=args.shoulder_lift_gain,
            elbow_flex_gain=args.elbow_flex_gain,
            wrist_flex_gain=args.wrist_flex_gain,
            wrist_roll_gain=args.wrist_roll_gain,
            gripper_open=args.gripper_open,
            gripper_closed=args.gripper_closed,
            pinch_closed_ratio=args.pinch_closed_ratio,
            pinch_open_ratio=args.pinch_open_ratio,
            mirror_hand=_resolve_mirror_hand(args),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.enable_robot:
        if not args.robot_port:
            raise SystemExit("--robot-port is required with --enable-robot")
        if not args.robot_id:
            raise SystemExit("--robot-id is required with --enable-robot")
        validate_robot_port_matches_id(args.robot_port, args.robot_id)
        calibration_file = args.calibration_dir.expanduser().resolve() / f"{args.robot_id}.json"
        if not calibration_file.exists():
            raise SystemExit(f"Calibration file not found: {calibration_file}")
        if not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0:
            raise SystemExit("--max-relative-target must be finite and positive")


def _resolve_mirror_hand(args: argparse.Namespace) -> bool:
    if args.mirror_hand == "on":
        return True
    if args.mirror_hand == "off":
        return False
    return args.arm == "left"


def validate_robot_port_matches_id(robot_port: str, robot_id: str) -> None:
    port_serial = robot_port_serial_hint(robot_port)
    robot_serial = robot_id_serial_hint(robot_id)
    if port_serial is None or robot_serial is None or port_serial == robot_serial:
        return
    raise SystemExit(
        f"--robot-port appears to be for serial {port_serial!r}, but --robot-id is {robot_id!r}."
    )


def robot_port_serial_hint(robot_port: str) -> str | None:
    match = re.search(r"(?:^|[.])(?:usbmodem|usbserial)([-_A-Za-z0-9]+)$", Path(robot_port).name)
    if match is None:
        return None
    return match.group(1).lstrip("-_") or None


def robot_id_serial_hint(robot_id: str) -> str | None:
    prefix = "so101_"
    if not robot_id.startswith(prefix):
        return None
    serial = robot_id.removeprefix(prefix)
    return serial or None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_main_args.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop/main.py playground/teleop/tests/test_main_args.py
git commit -m "feat(teleop): add main.py argument parser and validation"
```

---

## Task 12: main.py — loop, key handling, integration

**Files:**
- Modify: `playground/teleop/main.py` (add `main()` and `run_loop()`)
- Create: `playground/teleop/tests/test_main_loop.py`

- [ ] **Step 1: Write a failing integration test for handle_neutral_capture / handle_sync_toggle / handle_backend_send**

Create `playground/teleop/tests/test_main_loop.py`:

```python
import pytest

import main
from teleop.pose_mapper import MappingConfig, TeleopMapper
from teleop.robot_backend import DryRunBackend
from teleop.safety import SafetyConfig, TargetFilter
from teleop.types import (
    ACTION_KEYS,
    CONTROLLED_KEYS,
    ArmSample,
    FilterResult,
    FreezeReason,
    HandSample,
    Landmark,
    PoseLandmark,
    RobotTargets,
    TeleopSample,
)


def baseline() -> RobotTargets:
    return RobotTargets(0.0, 0.0, 0.0, 0.0, 0.0, 80.0)


def safety_config() -> SafetyConfig:
    return SafetyConfig(
        limits={
            "shoulder_pan.pos": (-20.0, 20.0),
            "shoulder_lift.pos": (-20.0, 20.0),
            "elbow_flex.pos": (-25.0, 25.0),
            "wrist_flex.pos": (-15.0, 15.0),
            "wrist_roll.pos": (-25.0, 25.0),
            "gripper.pos": (15.0, 85.0),
        },
        max_delta={key: 2.0 for key in CONTROLLED_KEYS},
        smoothing=0.35,
        stale_timeout_ms=200,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
    )


def mapping_config() -> MappingConfig:
    return MappingConfig(
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


def sample(timestamp_ms: int = 1000) -> TeleopSample:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        elbow=PoseLandmark(0.0, 0.3, 0.0, 0.9),
        wrist=PoseLandmark(0.0, 0.6, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=timestamp_ms,
    )
    points = [Landmark(0.5, 0.65, 0.0) for _ in range(21)]
    points[5] = Landmark(0.45, 0.55, 0.0)
    points[17] = Landmark(0.55, 0.55, 0.0)
    points[9] = Landmark(0.50, 0.45, 0.0)
    points[4] = Landmark(0.42, 0.40, 0.0)
    points[8] = Landmark(0.58, 0.40, 0.0)
    hand = HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=timestamp_ms)
    return TeleopSample(arm=arm, hand=hand, timestamp_ms=timestamp_ms)


def test_sample_is_usable_returns_true_for_good_sample() -> None:
    assert main.sample_is_usable(
        sample(),
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_pose_visibility_below_threshold() -> None:
    sample_with_low_visibility = sample()
    bad_arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.2),
        elbow=sample_with_low_visibility.arm.elbow,
        wrist=sample_with_low_visibility.arm.wrist,
        wrist_image_xy=sample_with_low_visibility.arm.wrist_image_xy,
        timestamp_ms=sample_with_low_visibility.arm.timestamp_ms,
    )
    s = TeleopSample(arm=bad_arm, hand=sample_with_low_visibility.hand, timestamp_ms=1000)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_arm_missing() -> None:
    s = TeleopSample(arm=None, hand=sample().hand, timestamp_ms=1000)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_sample_is_usable_false_when_stale() -> None:
    assert not main.sample_is_usable(
        sample(timestamp_ms=500),
        now_ms=1000,
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )


def test_handle_neutral_capture_rejects_unusable_sample() -> None:
    mapper = TeleopMapper(mapping_config())
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState()
    result = main.handle_neutral_capture(
        sample=TeleopSample(arm=None, hand=None, timestamp_ms=1000),
        now_ms=1000,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=baseline(),
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
        state=state,
    )
    assert result is target_filter
    assert mapper.neutral_ready is False
    assert state.notice is not None and "neutral rejected" in state.notice


def test_handle_neutral_capture_succeeds_with_usable_sample() -> None:
    mapper = TeleopMapper(mapping_config())
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState()
    result = main.handle_neutral_capture(
        sample=sample(),
        now_ms=1000,
        mapper=mapper,
        target_filter=target_filter,
        baseline_targets=baseline(),
        min_pose_visibility=0.6,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
        state=state,
    )
    assert mapper.neutral_ready is True
    assert state.notice == "neutral captured"


def test_handle_sync_toggle_disabled_when_send_failed() -> None:
    state = main.LoopState(send_failed=True)
    main.handle_sync_toggle(state)
    assert state.sync_enabled is False
    assert "send failed" in (state.notice or "")


def test_handle_sync_toggle_flips_when_send_ok() -> None:
    state = main.LoopState()
    main.handle_sync_toggle(state)
    assert state.sync_enabled is True
    main.handle_sync_toggle(state)
    assert state.sync_enabled is False


def test_handle_backend_send_locks_off_on_failure() -> None:
    class BrokenBackend(DryRunBackend):
        def send(self, _targets):
            raise RuntimeError("port closed")

    backend = BrokenBackend(default_gripper=80.0)
    backend.connect()
    target_filter = TargetFilter(safety_config(), baseline())
    state = main.LoopState(sync_enabled=True)
    result = FilterResult(
        targets=baseline(),
        frozen=False,
        clamped_keys=(),
        reason=FreezeReason.ACTIVE,
    )
    new_result = main.handle_backend_send(backend, result, target_filter, state)
    assert state.sync_enabled is False
    assert state.send_failed is True
    assert new_result.frozen is True
    assert new_result.reason is FreezeReason.PAUSED
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop && uv run pytest tests/test_main_loop.py -v
```

Expected: All tests FAIL with `AttributeError: module 'main' has no attribute 'LoopState'`.

- [ ] **Step 3: Implement LoopState + helpers + run_loop + main**

Append to `playground/teleop/main.py`:

```python
import time
from dataclasses import dataclass

import cv2

from teleop.pose_mapper import MappingConfig, TeleopMapper
from teleop.robot_backend import DryRunBackend, SO101Backend, SO101BackendConfig
from teleop.safety import SafetyConfig, TargetFilter
from teleop.tracker import (
    HAND_MODEL_URL,
    LatestHandResult,
    LatestPoseResult,
    POSE_MODEL_URL,
    POSE_LEFT_ELBOW,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_RIGHT_ELBOW,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
    create_hand_landmarker,
    create_pose_landmarker,
    draw_overlay,
    ensure_model,
    frame_to_mp_image,
    fuse_samples,
    open_camera,
)
from teleop.types import (
    CONTROLLED_KEYS,
    FilterResult,
    FreezeReason,
    RobotTargets,
    TeleopSample,
)


@dataclass
class LoopState:
    sync_enabled: bool = False
    notice: str | None = None
    send_failed: bool = False


def main() -> None:
    args = parse_args()
    apply_camera_defaults(args)
    pose_model = ensure_model(args.pose_model_path, url=POSE_MODEL_URL)
    hand_model = ensure_model(args.hand_model_path, url=HAND_MODEL_URL)
    validate_args(args)
    if args.check:
        print(f"Pose model ready: {pose_model}")
        print(f"Hand model ready: {hand_model}")
        if args.enable_robot:
            check_robot_imports(args)
            print("Robot config validated.")
        return

    backend = make_backend(args)
    capture = None
    try:
        capture = open_camera(args.camera_index, args.width, args.height)
        backend.connect()
        mapper = TeleopMapper(make_mapping_config(args))
        target_filter = TargetFilter(
            make_safety_config(args, backend.baseline_targets), backend.baseline_targets
        )

        latest_pose = LatestPoseResult()
        latest_hand = LatestHandResult()
        with create_pose_landmarker(
            model_path=pose_model,
            detection_confidence=args.detection_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            result_callback=latest_pose.update,
        ) as pose_landmarker, create_hand_landmarker(
            model_path=hand_model,
            max_hands=args.max_hands,
            detection_confidence=args.detection_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            result_callback=latest_hand.update,
        ) as hand_landmarker:
            run_loop(
                args=args,
                capture=capture,
                pose_landmarker=pose_landmarker,
                hand_landmarker=hand_landmarker,
                latest_pose=latest_pose,
                latest_hand=latest_hand,
                mapper=mapper,
                target_filter=target_filter,
                backend=backend,
            )
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        backend.disconnect()


def check_robot_imports(args) -> None:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir.expanduser().resolve(),
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    _ = SO101Follower


def make_backend(args):
    if not args.enable_robot:
        return DryRunBackend(default_gripper=args.gripper_open)
    return SO101Backend(
        SO101BackendConfig(
            port=args.robot_port,
            robot_id=args.robot_id,
            calibration_dir=args.calibration_dir.expanduser().resolve(),
            max_relative_target=args.max_relative_target,
        )
    )


def make_mapping_config(args) -> MappingConfig:
    return MappingConfig(
        shoulder_pan_gain=args.shoulder_pan_gain,
        shoulder_lift_gain=args.shoulder_lift_gain,
        elbow_flex_gain=args.elbow_flex_gain,
        wrist_flex_gain=args.wrist_flex_gain,
        wrist_roll_gain=args.wrist_roll_gain,
        gripper_open=args.gripper_open,
        gripper_closed=args.gripper_closed,
        pinch_closed_ratio=args.pinch_closed_ratio,
        pinch_open_ratio=args.pinch_open_ratio,
        mirror_hand=_resolve_mirror_hand(args),
    )


def make_safety_config(args, baseline: RobotTargets) -> SafetyConfig:
    if args.gripper_min >= args.gripper_max:
        raise ValueError("--gripper-min must be less than --gripper-max")
    return SafetyConfig(
        limits={
            "shoulder_pan.pos": (
                baseline.shoulder_pan - args.shoulder_pan_limit,
                baseline.shoulder_pan + args.shoulder_pan_limit,
            ),
            "shoulder_lift.pos": (
                baseline.shoulder_lift - args.shoulder_lift_limit,
                baseline.shoulder_lift + args.shoulder_lift_limit,
            ),
            "elbow_flex.pos": (
                baseline.elbow_flex - args.elbow_flex_limit,
                baseline.elbow_flex + args.elbow_flex_limit,
            ),
            "wrist_flex.pos": (
                baseline.wrist_flex - args.wrist_flex_limit,
                baseline.wrist_flex + args.wrist_flex_limit,
            ),
            "wrist_roll.pos": (
                baseline.wrist_roll - args.wrist_roll_limit,
                baseline.wrist_roll + args.wrist_roll_limit,
            ),
            "gripper.pos": (
                min(args.gripper_min, baseline.gripper),
                max(args.gripper_max, baseline.gripper),
            ),
        },
        max_delta={key: args.max_delta for key in CONTROLLED_KEYS},
        smoothing=args.smoothing,
        stale_timeout_ms=args.stale_timeout_ms,
        min_pose_visibility=args.min_pose_visibility,
        min_hand_confidence=args.min_hand_confidence,
    )


def run_loop(
    *,
    args,
    capture,
    pose_landmarker,
    hand_landmarker,
    latest_pose: LatestPoseResult,
    latest_hand: LatestHandResult,
    mapper: TeleopMapper,
    target_filter: TargetFilter,
    backend,
) -> None:
    start_time = time.monotonic()
    previous_timestamp_ms = -1
    previous_frame_time = start_time
    fps_display = 0.0
    state = LoopState()
    last_deadman_ms = -1
    last_result = FilterResult(
        target_filter.last_targets,
        frozen=True,
        clamped_keys=(),
        reason=FreezeReason.PAUSED,
    )
    consecutive_empty_frames = 0
    max_consecutive_empty_frames = 30

    while True:
        loop_start = time.monotonic()
        ok, frame = capture.read()
        if not ok:
            consecutive_empty_frames += 1
            if consecutive_empty_frames > max_consecutive_empty_frames:
                raise RuntimeError(
                    f"Camera returned {consecutive_empty_frames} empty frames in a row."
                )
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            time.sleep(1.0 / max(args.fps, 1))
            continue
        consecutive_empty_frames = 0
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        mp_image = frame_to_mp_image(frame)
        pose_landmarker.detect_async(mp_image, timestamp_ms)
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        arm_sample = latest_pose.best_arm_sample(arm=args.arm)
        hand_samples = latest_hand.all_hand_samples()
        teleop_sample = fuse_samples(
            arm=arm_sample, hands=hand_samples, timestamp_ms=timestamp_ms
        )

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            handle_sync_toggle(state)
        if key == ord("n"):
            target_filter = handle_neutral_capture(
                sample=teleop_sample,
                now_ms=timestamp_ms,
                mapper=mapper,
                target_filter=target_filter,
                baseline_targets=backend.baseline_targets,
                min_pose_visibility=args.min_pose_visibility,
                min_hand_confidence=args.min_hand_confidence,
                stale_timeout_ms=args.stale_timeout_ms,
                state=state,
            )
        if args.deadman_key and key == ord(args.deadman_key):
            last_deadman_ms = timestamp_ms

        deadman_active = not args.deadman_key or timestamp_ms - last_deadman_ms <= args.deadman_grace_ms
        tracking_ok = sample_is_usable(
            teleop_sample,
            now_ms=timestamp_ms,
            min_pose_visibility=args.min_pose_visibility,
            min_hand_confidence=args.min_hand_confidence,
            stale_timeout_ms=args.stale_timeout_ms,
        )
        desired = target_filter.last_targets
        if tracking_ok and mapper.neutral_ready:
            try:
                desired = mapper.map(teleop_sample)
            except ValueError:
                tracking_ok = False

        sample_ts = teleop_sample.timestamp_ms if teleop_sample.arm is not None else None
        try:
            last_result = target_filter.update(
                desired,
                now_ms=timestamp_ms,
                sample_timestamp_ms=sample_ts,
                sync_enabled=state.sync_enabled,
                neutral_ready=mapper.neutral_ready,
                deadman_active=deadman_active,
                tracking_ok=tracking_ok,
            )
        except ValueError:
            last_result = FilterResult(
                target_filter.last_targets,
                frozen=True,
                clamped_keys=(),
                reason=FreezeReason.TRACKING_LOST,
            )
        last_result = handle_backend_send(backend, last_result, target_filter, state)

        now = time.monotonic()
        instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
        fps_display = (
            instantaneous_fps if fps_display == 0.0 else (0.9 * fps_display) + (0.1 * instantaneous_fps)
        )
        previous_frame_time = now

        status_lines = build_status_lines(
            args=args,
            fps_display=fps_display,
            sync_enabled=state.sync_enabled,
            neutral_ready=mapper.neutral_ready,
            sample=teleop_sample,
            result=last_result,
            notice=state.notice,
        )
        arm_image_landmarks = _arm_image_landmarks(args.arm, teleop_sample)
        draw_overlay(
            frame,
            arm=teleop_sample.arm,
            hand=teleop_sample.hand,
            status_lines=status_lines,
            image_size=(args.width, args.height),
            arm_image_landmarks=arm_image_landmarks,
        )
        cv2.imshow(WINDOW_NAME, frame)

        elapsed = time.monotonic() - loop_start
        sleep_s = max((1.0 / args.fps) - elapsed, 0.0)
        if sleep_s:
            time.sleep(sleep_s)


def _arm_image_landmarks(
    arm: str, sample: TeleopSample
) -> dict[str, tuple[float, float]] | None:
    if sample.arm is None:
        return None
    return {"wrist": sample.arm.wrist_image_xy}


def sample_is_usable(
    sample: TeleopSample,
    *,
    now_ms: int,
    min_pose_visibility: float,
    min_hand_confidence: float,
    stale_timeout_ms: int,
) -> bool:
    if sample.arm is None or sample.hand is None:
        return False
    if sample.arm.shoulder.visibility < min_pose_visibility:
        return False
    if sample.arm.elbow.visibility < min_pose_visibility:
        return False
    if sample.arm.wrist.visibility < min_pose_visibility:
        return False
    if sample.hand.confidence < min_hand_confidence:
        return False
    if now_ms - sample.timestamp_ms > stale_timeout_ms:
        return False
    return True


def handle_sync_toggle(state: LoopState) -> None:
    if state.send_failed:
        state.sync_enabled = False
        state.notice = "sync locked off: send failed"
        return
    state.sync_enabled = not state.sync_enabled
    if state.sync_enabled:
        state.notice = None


def handle_neutral_capture(
    *,
    sample: TeleopSample,
    now_ms: int,
    mapper: TeleopMapper,
    target_filter: TargetFilter,
    baseline_targets: RobotTargets,
    min_pose_visibility: float,
    min_hand_confidence: float,
    stale_timeout_ms: int,
    state: LoopState,
) -> TargetFilter:
    if not sample_is_usable(
        sample,
        now_ms=now_ms,
        min_pose_visibility=min_pose_visibility,
        min_hand_confidence=min_hand_confidence,
        stale_timeout_ms=stale_timeout_ms,
    ):
        state.notice = "neutral rejected: tracking degraded"
        return target_filter

    try:
        mapper.capture_neutral(sample, baseline_targets)
        state.notice = "neutral captured"
        return TargetFilter(target_filter.config, baseline_targets)
    except ValueError as exc:
        state.notice = f"neutral rejected: {exc}"
        return target_filter


def handle_backend_send(
    backend, result: FilterResult, target_filter: TargetFilter, state: LoopState
) -> FilterResult:
    if state.send_failed:
        return FilterResult(
            target_filter.last_targets,
            frozen=True,
            clamped_keys=(),
            reason=FreezeReason.PAUSED,
        )
    if result.frozen:
        return result

    try:
        backend.send(result.targets)
        return result
    except Exception as exc:
        state.sync_enabled = False
        state.send_failed = True
        state.notice = f"send failed: {exc}"
        return FilterResult(
            target_filter.last_targets,
            frozen=True,
            clamped_keys=(),
            reason=FreezeReason.PAUSED,
        )


def build_status_lines(
    *,
    args,
    fps_display: float,
    sync_enabled: bool,
    neutral_ready: bool,
    sample: TeleopSample,
    result: FilterResult,
    notice: str | None,
) -> list[str]:
    robot_state = "ROBOT" if args.enable_robot else "DRY"
    arm_state = "none"
    if sample.arm is not None:
        arm_state = (
            f"{args.arm} s={sample.arm.shoulder.visibility:.2f} "
            f"e={sample.arm.elbow.visibility:.2f} w={sample.arm.wrist.visibility:.2f}"
        )
    hand_state = "none" if sample.hand is None else f"{sample.hand.handedness} {sample.hand.confidence:.2f}"
    clamp_text = ",".join(result.clamped_keys) if result.clamped_keys else "none"
    targets = result.targets
    line1 = (
        f"{robot_state} | {fps_display:4.1f} FPS | sync={'on' if sync_enabled else 'off'} | "
        f"neutral={'yes' if neutral_ready else 'no'} | reason={result.reason.value} | clamp={clamp_text}"
    )
    line2 = f"arm={arm_state} | hand={hand_state}"
    line3 = (
        f"pan={targets.shoulder_pan:.1f} lift={targets.shoulder_lift:.1f} "
        f"elb={targets.elbow_flex:.1f} wf={targets.wrist_flex:.1f} "
        f"wr={targets.wrist_roll:.1f} grip={targets.gripper:.1f}"
    )
    lines = [line1, line2, line3]
    if notice:
        lines.append(f"notice: {notice}")
    return lines


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd playground/teleop && uv run pytest tests/test_main_loop.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Run the full suite**

```bash
cd playground/teleop && uv run pytest -v
```

Expected: All tests across all files PASS.

- [ ] **Step 6: Commit**

```bash
git add playground/teleop/main.py playground/teleop/tests/test_main_loop.py
git commit -m "feat(teleop): wire main loop, key handling, and overlay"
```

---

## Task 13: Smoke `--check` end-to-end on real binary

**Files:**
- Modify: `playground/teleop/main.py` (only if smoke uncovers a bug — otherwise no code change)

- [ ] **Step 1: Run --check and confirm both models download and validation passes**

```bash
cd playground/teleop && uv run python main.py --check
```

Expected output (first run downloads, subsequent runs skip):

```
Downloading MediaPipe model to .../playground/teleop/models/pose_landmarker_lite.task
Downloading MediaPipe model to .../playground/teleop/models/hand_landmarker.task
Pose model ready: .../pose_landmarker_lite.task
Hand model ready: .../hand_landmarker.task
```

Exit code 0.

- [ ] **Step 2: If models are downloaded successfully and the command exits 0, no code change is needed**

If a bug is uncovered, return to whichever earlier task introduced it, fix with a TDD step, re-run the suite, and commit separately.

- [ ] **Step 3: Commit any incidental fixes (skip if none)**

```bash
git add -p   # only the relevant files
git commit -m "fix(teleop): <describe>"
```

---

## Task 14: README with setup, controls, and safety guidance

**Files:**
- Modify: `playground/teleop/README.md`

- [ ] **Step 1: Replace the stub README**

Overwrite `playground/teleop/README.md`:

````markdown
# Egocentric SO101 6-DOF Teleop

Egocentric webcam teleoperation prototype for the SO101 follower. A chest- or
torso-mounted webcam looks forward and down at your arm; MediaPipe Pose +
Hand Landmarker drive all six follower joints.

This is the egocentric companion to `playground/mediapipe_so101/`, which uses an
exocentric front-facing camera and controls only the wrist and gripper.

## Setup

```bash
uv sync
```

## Check

Validates the model downloads and (optionally) the robot config without opening
the camera:

```bash
uv run python main.py --check
```

## Dry Run

```bash
uv run python main.py --camera-index 0 --fps 15
```

Controls:

- `n`: capture neutral arm + hand pose. Requires both the Pose arm and Hand
  Landmarker to be giving high-confidence results simultaneously.
- `space`: toggle real-time sync.
- `q` or `Esc`: exit.

Before neutral capture, no targets are emitted.

## Robot Mode

Validate robot configuration without opening the camera:

```bash
uv run python main.py \
  --check \
  --enable-robot \
  --robot-port /dev/cu.usbmodemYOUR_PORT \
  --robot-id so101_YOUR_ID \
  --calibration-dir ../so101/calibration/robots/so_follower
```

First physical run (always start with deadman + conservative limits):

```bash
uv run python main.py \
  --enable-robot \
  --robot-port /dev/cu.usbmodemYOUR_PORT \
  --robot-id so101_YOUR_ID \
  --calibration-dir ../so101/calibration/robots/so_follower \
  --fps 10 \
  --max-delta 2.0 \
  --shoulder-pan-limit 15 \
  --shoulder-lift-limit 15 \
  --elbow-flex-limit 20 \
  --wrist-flex-limit 10 \
  --wrist-roll-limit 20 \
  --deadman-key x
```

## Camera mounting

Designed for a chest- or torso-mounted webcam looking forward and slightly down.
Your forearm and hand should always be in frame; your shoulder is often
partially visible. The visibility numbers next to each pose landmark on the
overlay help tune `--min-pose-visibility` for your specific mounting.

## Mapping

The arm joints are mapped from MediaPipe Pose's `pose_world_landmarks` (3D,
metric, hip-centered):

- `shoulder_pan.pos` from the yaw of the upper arm vector around the body axis.
- `shoulder_lift.pos` from the pitch of the upper arm relative to vertical.
- `elbow_flex.pos` from the angle between upper arm and forearm.

The wrist and gripper come from the 2D Hand Landmarker, identical to the
`playground/mediapipe_so101/` exocentric playground:

- `wrist_roll.pos` from palm left-right tilt.
- `wrist_flex.pos` from hand flex relative to neutral.
- `gripper.pos` from continuous thumb-index pinch openness.

Arm and wrist features are reported as deltas from the neutral capture and
scaled by per-joint gain CLI flags. Gripper is absolute. The
`--mirror-hand {auto,on,off}` flag handles the egocentric back-of-hand view —
`auto` infers from `--arm`.

## Model Cache

The Pose and Hand `.task` files are downloaded on first use into `models/` and
ignored by git.

## Safety Notes

This commands all six SO101 follower joints. Always:

- Start with `--deadman-key x` and conservative limits.
- Keep the robot's workspace clear; the shoulder and elbow have a much larger
  swept volume than the wrist alone.
- Watch the visibility scores on the overlay — if `--min-pose-visibility` is
  too low, intermittent landmark dropouts will hold-last instead of freezing.
- The script freezes command output when tracking is missing, stale, below
  confidence, paused, the deadman key is inactive, or neutral has not been
  captured.
- A backend send failure disables sync and locks command output off until
  restart.
````

- [ ] **Step 2: Commit**

```bash
git add playground/teleop/README.md
git commit -m "docs(teleop): add full README with setup, controls, safety"
```

---

## Self-Review (already performed)

- **Spec coverage** — every spec section maps to at least one task:
  - Directory layout → Task 1.
  - `types.py` → Task 2.
  - `WristMapper` (ported) → Task 3. `ArmMapper` (new math, all canonical poses + translation invariance) → Task 4. `TeleopMapper` → Task 5.
  - `SafetyConfig` + 6-joint `TargetFilter` + visibility threshold config → Task 6. Note: the visibility-based freeze is enforced by the loop calling `sample_is_usable` and passing `tracking_ok=False`, validated in Task 12's `test_sample_is_usable_false_when_pose_visibility_below_threshold`.
  - `DryRunBackend` / `SO101Backend` (6 joints, mocked) → Task 7.
  - Model paths + ensure_model → Task 8. Pose + Hand landmarker creation + fusion → Task 9. Overlay → Task 10.
  - `main.py` argparse + validation → Task 11. Loop + key handling + integration → Task 12. End-to-end `--check` smoke → Task 13.
  - README → Task 14.

- **Placeholder scan** — no TBD/TODO/"implement later"; every step has either code or a concrete command with expected output.

- **Type consistency** — `RobotTargets` carries six fields in the canonical order `(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)` throughout; `ACTION_KEYS` and `CONTROLLED_KEYS` use the matching `.pos` suffix everywhere; `MappingConfig` field names (`shoulder_pan_gain`, etc.) match across tests, mapper code, and main.py.

---

## Execution Handoff

(See parent harness for handoff prompt.)
