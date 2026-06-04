# Egocentric SO101 5-DOF Hand-Only Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `playground/teleop2/` — a hand-only egocentric teleoperation prototype that controls 5 SO101 follower joints (holding `elbow_flex` at baseline) using only MediaPipe Hand Landmarker output. The wrist landmark's image-plane x position drives `shoulder_pan`; the apparent hand size drives `shoulder_lift`; the existing exocentric hand math drives `wrist_flex`, `wrist_roll`, and `gripper`.

**Architecture:** Fork the existing `playground/teleop/` (which already has 6-DOF safety/backend infrastructure plus a proven `WristMapper`) and strip out everything Pose-related (Pose Landmarker, ArmSample, ArmMapper, fusion, pose-visibility safety, pose skeleton overlay). Then add a new `HandPositionMapper` that derives `shoulder_pan`/`shoulder_lift` from the hand wrist landmark's image-plane position and the hand's apparent size (an existing measurement). Rewire `TeleopMapper` to compose `WristMapper` + `HandPositionMapper` and hold `elbow_flex` at baseline.

**Tech Stack:** Python 3.12, uv, MediaPipe 0.10+ (Hand Landmarker only — Pose is in the same package but not imported), OpenCV, LeRobot 0.5.1 (`SO101Follower`), pytest. Same dependency surface as `playground/teleop/`.

---

## File Structure

```
playground/teleop2/
├── README.md                       # Setup, dry-run/robot commands, controls, camera angle guidance, safety
├── pyproject.toml                  # uv project, same deps as teleop
├── main.py                         # argparse, validate, camera loop, key handling, overlay
└── teleop2/                        # package dir (renamed from "teleop" during fork)
    ├── __init__.py
    ├── types.py                    # Landmark, HandSample, TeleopSample (hand-only), RobotTargets,
    │                               # FilterResult, FreezeReason, ACTION_KEYS, CONTROLLED_KEYS
    ├── tracker.py                  # Hand model download/creation, camera open,
    │                               # LatestHandResult, frame_to_mp_image, draw_overlay
    ├── pose_mapper.py              # MappingConfig, WristMapper, HandPositionMapper, TeleopMapper
    ├── safety.py                   # SafetyConfig (no min_pose_visibility), TargetFilter (6 joints)
    └── robot_backend.py            # Backend Protocol, DryRunBackend, SO101Backend (6 joints, unchanged)
tests/
    ├── __init__.py
    ├── test_smoke.py
    ├── test_types.py
    ├── test_pose_mapper.py
    ├── test_safety.py
    ├── test_robot_backend.py
    ├── test_tracker.py
    ├── test_main_args.py
    └── test_main_loop.py
```

The implementation strategy is **fork-and-modify** rather than build-from-scratch. The first task copies `playground/teleop/` as the starting baseline; subsequent tasks delete the Pose machinery and add the new `HandPositionMapper`. This is faster, type-consistent, and reuses the proven 6-DOF safety/backend code unchanged.

---

## Task 1: Fork `playground/teleop/` into `playground/teleop2/`

**Files:**
- Copy: every file under `playground/teleop/` to `playground/teleop2/`
- Rename: `playground/teleop2/teleop/` → `playground/teleop2/teleop2/`
- Modify: `playground/teleop2/pyproject.toml` (project name)
- Modify: every Python file with `from teleop.X import ...` or `import teleop` (change to `teleop2`)

- [ ] **Step 1: Copy the whole playground**

```bash
cp -r playground/teleop playground/teleop2
```

- [ ] **Step 2: Rename the package directory**

```bash
git mv playground/teleop2/teleop playground/teleop2/teleop2
```

Wait — `cp` doesn't stage anything in git yet, so `git mv` of a path under `teleop2` will fail because git doesn't know `playground/teleop2/teleop/` exists. Use plain `mv`:

```bash
mv playground/teleop2/teleop playground/teleop2/teleop2
```

- [ ] **Step 3: Update the project name in pyproject.toml**

In `playground/teleop2/pyproject.toml`, change `name = "teleop"` to `name = "teleop2"` and the description to `"Egocentric SO101 5-DOF hand-only teleoperation prototype using MediaPipe Hand."` Leave the dependencies and pytest config block unchanged.

- [ ] **Step 4: Update all imports from `teleop` → `teleop2`**

Replace `from teleop.` with `from teleop2.` and `import teleop` with `import teleop2` across every Python file under `playground/teleop2/`. Use:

```bash
find playground/teleop2 -name "*.py" -exec sed -i '' 's/from teleop\./from teleop2./g; s/^import teleop$/import teleop2/g' {} +
```

(`sed -i ''` is the macOS form; on Linux drop the empty quotes.)

The `tests/test_smoke.py` line `import teleop  # noqa: F401` becomes `import teleop2  # noqa: F401`.

- [ ] **Step 5: Delete the lock file and re-sync**

The lock file references `name = "teleop"`. Force a fresh sync:

```bash
rm playground/teleop2/uv.lock
cd playground/teleop2 && uv sync
```

- [ ] **Step 6: Run the full suite to verify the rename worked**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: all tests pass (≥104, including the two fix-up tests added at the end of the teleop session). If any tests fail with `ModuleNotFoundError: No module named 'teleop'`, you missed an import; fix and re-run.

- [ ] **Step 7: Update the README header**

In `playground/teleop2/README.md`, change the first line from `# Egocentric SO101 6-DOF Teleop` to `# Egocentric SO101 5-DOF Hand-Only Teleop` and the next paragraph to:

```markdown
Hand-only egocentric webcam teleoperation prototype for the SO101 follower. A slanted top-down webcam looks at the user's hand on or above a desk; MediaPipe Hand Landmarker drives five follower joints (shoulder_pan, shoulder_lift, wrist_flex, wrist_roll, gripper). `elbow_flex` is held at the operator's starting position.

This is the hand-only companion to `playground/teleop/`, which uses an egocentric chest-mounted camera with MediaPipe Pose + Hand to drive all six joints. We'll do a full README rewrite in the final task; the stub change here just keeps the doc honest.
```

- [ ] **Step 8: Commit**

```bash
git add playground/teleop2/
git commit -m "feat(teleop2): fork playground/teleop as starting point for hand-only variant"
```

---

## Task 2: Strip Pose Landmarker perception from tracker

**Files:**
- Modify: `playground/teleop2/teleop2/tracker.py`
- Modify: `playground/teleop2/tests/test_tracker.py`

- [ ] **Step 1: Delete Pose-related symbols from tracker.py**

Open `playground/teleop2/teleop2/tracker.py` and delete:

- The constant `POSE_MODEL_URL`.
- The function `default_pose_model_path()`.
- The six pose-index constants `POSE_RIGHT_SHOULDER`, `POSE_RIGHT_ELBOW`, `POSE_RIGHT_WRIST`, `POSE_LEFT_SHOULDER`, `POSE_LEFT_ELBOW`, `POSE_LEFT_WRIST`.
- The function `create_pose_landmarker(...)`.
- The class `LatestPoseResult` (constructor + `update` + `best_arm_sample`).
- The function `fuse_samples(...)`.
- The imports `ArmSample`, `PoseLandmark`, `TeleopSample` from the types import line (we'll add `TeleopSample` back later — for now `fuse_samples` is gone so we don't need it in `tracker.py` at all). The remaining types import becomes `from .types import HandSample, Landmark`.

Leave intact: `HAND_MODEL_URL`, `default_hand_model_path()`, `ensure_model(...)`, `open_camera(...)`, `frame_to_mp_image(...)`, `create_hand_landmarker(...)`, `LatestHandResult`, `_extract_best_hand`, `HAND_CONNECTIONS`, `draw_overlay(...)`. We will edit `draw_overlay` in a later task; the unused `arm_image_landmarks` parameter and the arm-drawing block become dead code temporarily — that's fine.

- [ ] **Step 2: Delete Pose-related tests from test_tracker.py**

Open `playground/teleop2/tests/test_tracker.py` and delete:

- The import line `from teleop2.tracker import (... LatestPoseResult, ..., fuse_samples, ...)` — keep only `LatestHandResult`, `default_hand_model_path`, `ensure_model`, plus `draw_overlay` if it's imported.
- The import `from teleop2.types import ArmSample, ..., PoseLandmark` — keep only `HandSample`, `Landmark` (and `PoseLandmark` if any remaining test still needs it — most likely none do).
- The helper `make_pose_landmark_obj`, `make_pose_result`.
- All `LatestPoseResult`-related tests (`test_latest_pose_result_returns_none_when_no_callback_received`, `test_latest_pose_result_returns_arm_sample_for_chosen_arm`, `test_latest_pose_result_supports_left_arm`).
- All `fuse_samples`-related tests (`test_fuse_samples_pairs_arm_with_nearest_hand_in_image_plane`, `test_fuse_samples_returns_none_hand_when_no_hands_present`, `test_fuse_samples_returns_none_arm_when_arm_missing`, `test_fuse_samples_uses_oldest_underlying_sample_timestamp`, `test_fuse_samples_uses_arm_timestamp_when_hand_missing`, `test_fuse_samples_uses_hand_timestamp_when_arm_missing`, `test_fuse_samples_falls_back_to_loop_clock_when_no_samples`).
- The test `test_best_arm_sample_populates_all_image_xy_fields`.
- The `default_pose_model_path` test: `test_default_pose_model_path_lives_under_models_dir`.

Some draw-overlay tests pass an `arm` parameter — leave those tests in place for now, the next task on the overlay will rewrite them.

- [ ] **Step 3: Run the tracker tests to confirm they pass**

```bash
cd playground/teleop2 && uv run pytest tests/test_tracker.py -v
```

Expected: All remaining tracker tests pass. If something fails because it imports a now-deleted symbol, edit the test file.

- [ ] **Step 4: Run the full suite — expect failures in other test files (they will be cleaned up in subsequent tasks)**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: tests in `test_pose_mapper.py`, `test_safety.py`, `test_main_args.py`, `test_main_loop.py`, `test_types.py` will fail because they still reference the deleted `ArmSample`, `PoseLandmark`, `fuse_samples`, etc. This is expected; subsequent tasks clean them up.

- [ ] **Step 5: Commit**

```bash
git add playground/teleop2/teleop2/tracker.py playground/teleop2/tests/test_tracker.py
git commit -m "feat(teleop2): strip Pose Landmarker plumbing from tracker"
```

---

## Task 3: Strip ArmSample and PoseLandmark from types

**Files:**
- Modify: `playground/teleop2/teleop2/types.py`
- Modify: `playground/teleop2/tests/test_types.py`

- [ ] **Step 1: Delete ArmSample and PoseLandmark from types.py**

In `playground/teleop2/teleop2/types.py`, delete the `ArmSample` and `PoseLandmark` dataclass definitions. Then modify the `TeleopSample` dataclass to drop the `arm` field:

```python
@dataclass(frozen=True)
class TeleopSample:
    hand: HandSample | None
    timestamp_ms: int
```

Leave everything else intact: `ACTION_KEYS`, `CONTROLLED_KEYS`, `Landmark`, `HandSample`, `RobotTargets`, `FreezeReason`, `FilterResult`.

- [ ] **Step 2: Update test_types.py**

In `playground/teleop2/tests/test_types.py`:

- Remove `ArmSample` and `PoseLandmark` from the imports list at the top.
- Delete the test `test_pose_landmark_carries_visibility`.
- Delete the test `test_arm_sample_holds_shoulder_elbow_wrist_pose_landmarks_and_timestamp`.
- Modify `test_teleop_sample_holds_arm_and_hand` so it no longer references arm — rename it to `test_teleop_sample_holds_hand` and simplify:

```python
def test_teleop_sample_holds_hand() -> None:
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.8,
        timestamp_ms=20,
    )
    sample = TeleopSample(hand=hand, timestamp_ms=20)
    assert sample.hand is hand
    assert sample.timestamp_ms == 20
```

- Replace `test_teleop_sample_allows_missing_hand_or_arm` with:

```python
def test_teleop_sample_allows_missing_hand() -> None:
    sample = TeleopSample(hand=None, timestamp_ms=20)
    assert sample.hand is None
    assert sample.timestamp_ms == 20
```

- [ ] **Step 3: Run test_types.py to verify**

```bash
cd playground/teleop2 && uv run pytest tests/test_types.py -v
```

Expected: All types tests pass.

- [ ] **Step 4: Commit**

```bash
git add playground/teleop2/teleop2/types.py playground/teleop2/tests/test_types.py
git commit -m "feat(teleop2): drop ArmSample/PoseLandmark; TeleopSample is hand-only"
```

---

## Task 4: Strip ArmMapper from pose_mapper

**Files:**
- Modify: `playground/teleop2/teleop2/pose_mapper.py`
- Modify: `playground/teleop2/tests/test_pose_mapper.py`

This task removes the arm-side mapping and stubs `TeleopMapper` so the file still imports. Task 5 adds the real `HandPositionMapper`; Task 6 rewires `TeleopMapper` to compose it.

- [ ] **Step 1: Delete arm-side symbols from pose_mapper.py**

In `playground/teleop2/teleop2/pose_mapper.py`, delete:

- The dataclass `ArmFeatures`.
- The function `extract_arm_features(...)`.
- The function `_validate_pose_landmark(...)`.
- The class `ArmMapper`.

Update the consolidated import at the top: change `from .types import ArmSample, HandSample, Landmark, PoseLandmark, RobotTargets, TeleopSample` to `from .types import HandSample, Landmark, RobotTargets, TeleopSample`.

Also delete these unused fields from `MappingConfig` (they were for the arm side):

- `shoulder_pan_gain`
- `shoulder_lift_gain`
- `elbow_flex_gain`
- `min_arm_segment`

And remove these names from the `numeric_fields` tuple inside `MappingConfig.__post_init__`, and remove the `if self.min_arm_segment <= 0.0: raise ...` check.

The remaining MappingConfig should be:

```python
@dataclass(frozen=True)
class MappingConfig:
    wrist_flex_gain: float = 30.0
    wrist_roll_gain: float = 60.0
    gripper_open: float = 80.0
    gripper_closed: float = 20.0
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.40
    min_hand_width: float = 0.03
    mirror_hand: bool = False

    def __post_init__(self) -> None:
        numeric_fields = (
            "wrist_flex_gain",
            "wrist_roll_gain",
            "gripper_open",
            "gripper_closed",
            "pinch_closed_ratio",
            "pinch_open_ratio",
            "min_hand_width",
        )
        for field_name in numeric_fields:
            _validate_finite_number(field_name, getattr(self, field_name))

        if self.min_hand_width <= 0.0:
            raise ValueError("min_hand_width must be greater than 0")
        if self.pinch_open_ratio <= self.pinch_closed_ratio:
            raise ValueError("pinch_open_ratio must be greater than pinch_closed_ratio")
```

Task 5 will add `shoulder_pan_gain`, `shoulder_lift_gain`, and `invert_shoulder_lift` back to `MappingConfig`. For now keep it minimal.

- [ ] **Step 2: Stub TeleopMapper to compile**

In `playground/teleop2/teleop2/pose_mapper.py`, replace the `TeleopMapper` class with this stub. Task 6 rewires it properly; for now it just composes `WristMapper` and holds the other joints at baseline.

```python
class TeleopMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.wrist = WristMapper(config)
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self.wrist.neutral_ready and self._neutral_targets is not None

    def capture_neutral(self, sample: TeleopSample, robot_targets: RobotTargets) -> None:
        if sample.hand is None:
            raise ValueError("TeleopMapper neutral capture requires a hand sample")
        self.wrist.capture_neutral(sample.hand, robot_targets)
        self._neutral_targets = robot_targets

    def map(self, sample: TeleopSample) -> RobotTargets:
        if sample.hand is None:
            raise ValueError("TeleopMapper map requires a hand sample")
        if self._neutral_targets is None:
            raise RuntimeError("Neutral TeleopMapper sample has not been captured")

        wrist_targets = self.wrist.map(sample.hand)
        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan,
            shoulder_lift=self._neutral_targets.shoulder_lift,
            elbow_flex=self._neutral_targets.elbow_flex,
            wrist_flex=wrist_targets.wrist_flex,
            wrist_roll=wrist_targets.wrist_roll,
            gripper=wrist_targets.gripper,
        )
```

- [ ] **Step 3: Strip arm-related test code from test_pose_mapper.py**

In `playground/teleop2/tests/test_pose_mapper.py`:

- Remove `ArmMapper`, `ArmSample`, `PoseLandmark`, `extract_arm_features` from imports.
- Delete the `arm()` helper function and the `arm_mapper()` helper.
- Delete all ArmMapper tests (any test function with "arm" in the name except for `test_extract_wrist_features_*`).
- Delete the `_validate_pose_landmark` reference if any test imports it directly.
- Modify `wrist_mapper()` helper to drop the arm-related MappingConfig fields it sets. Updated helper:

```python
def wrist_mapper(**overrides) -> WristMapper:
    defaults = dict(
        wrist_flex_gain=30.0,
        wrist_roll_gain=60.0,
        gripper_open=80.0,
        gripper_closed=20.0,
        pinch_closed_ratio=0.35,
        pinch_open_ratio=1.40,
        mirror_hand=False,
    )
    defaults.update(overrides)
    return WristMapper(MappingConfig(**defaults))
```

- Modify `baseline_targets()` to keep returning all 6 RobotTargets fields (unchanged from teleop's version).

- Update the `teleop_sample()` helper to drop the arm parameter:

```python
def teleop_sample(*, hand_sample=None, timestamp_ms=1000) -> TeleopSample:
    return TeleopSample(
        hand=hand_sample if hand_sample is not None else hand(),
        timestamp_ms=timestamp_ms,
    )
```

- Update the `teleop_mapper()` helper to drop arm-related MappingConfig fields:

```python
def teleop_mapper(**overrides) -> TeleopMapper:
    defaults = dict(
        wrist_flex_gain=30.0,
        wrist_roll_gain=60.0,
        gripper_open=80.0,
        gripper_closed=20.0,
        pinch_closed_ratio=0.35,
        pinch_open_ratio=1.40,
    )
    defaults.update(overrides)
    return TeleopMapper(MappingConfig(**defaults))
```

- Update the TeleopMapper tests to use the new shape. Replace the existing `test_teleop_mapper_neutral_requires_both_arm_and_hand` with:

```python
def test_teleop_mapper_neutral_requires_hand() -> None:
    mapper = teleop_mapper()

    with pytest.raises(ValueError, match="requires a hand sample"):
        mapper.capture_neutral(
            TeleopSample(hand=None, timestamp_ms=1000), baseline_targets()
        )
```

And update `test_teleop_mapper_emits_six_dof_targets_after_neutral_capture` to:

```python
def test_teleop_mapper_emits_six_dof_targets_after_neutral_capture() -> None:
    mapper = teleop_mapper()
    sample = teleop_sample()
    mapper.capture_neutral(sample, baseline_targets())

    targets = mapper.map(sample)

    # shoulder_pan / lift / elbow_flex are held at baseline (Task 5/6 will add live shoulder control)
    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan)
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift)
    assert targets.elbow_flex == pytest.approx(baseline_targets().elbow_flex)
    assert targets.wrist_flex == pytest.approx(baseline_targets().wrist_flex)
    assert targets.wrist_roll == pytest.approx(baseline_targets().wrist_roll)
    assert targets.gripper == pytest.approx(80.0)
```

And update `test_teleop_mapper_neutral_ready_requires_both_sides` to:

```python
def test_teleop_mapper_neutral_ready_requires_hand_captured() -> None:
    mapper = teleop_mapper()
    assert mapper.neutral_ready is False

    mapper.capture_neutral(teleop_sample(), baseline_targets())
    assert mapper.neutral_ready is True
```

And update `test_teleop_mapper_map_requires_arm_and_hand_present` to:

```python
def test_teleop_mapper_map_requires_hand_present() -> None:
    mapper = teleop_mapper()
    mapper.capture_neutral(teleop_sample(), baseline_targets())

    with pytest.raises(ValueError, match="requires a hand sample"):
        mapper.map(TeleopSample(hand=None, timestamp_ms=1000))
```

Keep `test_teleop_mapper_map_requires_neutral_capture` unchanged.

- [ ] **Step 4: Run pose_mapper tests to verify**

```bash
cd playground/teleop2 && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All remaining pose_mapper tests pass (wrist tests still all pass; the trimmed TeleopMapper tests pass).

- [ ] **Step 5: Commit**

```bash
git add playground/teleop2/teleop2/pose_mapper.py playground/teleop2/tests/test_pose_mapper.py
git commit -m "feat(teleop2): strip ArmMapper; stub TeleopMapper to hold all 3 arm joints at baseline"
```

---

## Task 5: Strip min_pose_visibility from safety and main

**Files:**
- Modify: `playground/teleop2/teleop2/safety.py`
- Modify: `playground/teleop2/tests/test_safety.py`
- Modify: `playground/teleop2/main.py`
- Modify: `playground/teleop2/tests/test_main_args.py`
- Modify: `playground/teleop2/tests/test_main_loop.py`

- [ ] **Step 1: Drop min_pose_visibility from SafetyConfig**

In `playground/teleop2/teleop2/safety.py`, delete the `min_pose_visibility` field and its `__post_init__` validation block. The remaining `SafetyConfig` is:

```python
@dataclass(frozen=True)
class SafetyConfig:
    limits: Mapping[str, tuple[float, float]]
    max_delta: Mapping[str, float]
    smoothing: float
    stale_timeout_ms: int
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
        if (
            not math.isfinite(self.min_hand_confidence)
            or not 0.0 <= self.min_hand_confidence <= 1.0
        ):
            raise ValueError("min_hand_confidence must be in [0, 1]")

        object.__setattr__(self, "limits", MappingProxyType(validated_limits))
        object.__setattr__(self, "max_delta", MappingProxyType(validated_max_delta))
```

`TargetFilter` itself doesn't need any changes — it never read `min_pose_visibility`.

- [ ] **Step 2: Update test_safety.py**

In `playground/teleop2/tests/test_safety.py`:

- Update the `safety_config()` helper to drop `min_pose_visibility` from defaults.
- Remove `min_pose_visibility` from any `SafetyConfig(...)` constructor call in the test file.

Updated helper:

```python
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
        min_hand_confidence=0.5,
    )
    defaults.update(overrides)
    return SafetyConfig(**defaults)
```

Same edit for any explicit `SafetyConfig(...)` invocations in `test_safety_config_requires_*` tests — drop the `min_pose_visibility` kwarg.

- [ ] **Step 3: Drop --min-pose-visibility from main.py**

In `playground/teleop2/main.py`:

- Remove the `--min-pose-visibility` arg in `parse_args`.
- Remove the loop `for name in (... "min_pose_visibility")` entry in `validate_args` — actually, change the tuple to exclude `min_pose_visibility`:

```python
for name in (
    "detection_confidence",
    "presence_confidence",
    "tracking_confidence",
    "min_hand_confidence",
):
    value = getattr(args, name)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
```

- In `make_safety_config`, drop the `min_pose_visibility=args.min_pose_visibility` kwarg from the `SafetyConfig(...)` call.
- In `sample_is_usable`, remove the `min_pose_visibility` parameter and all three `sample.arm.shoulder.visibility / elbow.visibility / wrist.visibility < min_pose_visibility` checks. Also remove the `sample.arm is None` check (we don't have an arm anymore — only check hand). Updated:

```python
def sample_is_usable(
    sample: TeleopSample,
    *,
    now_ms: int,
    min_hand_confidence: float,
    stale_timeout_ms: int,
) -> bool:
    if sample.hand is None:
        return False
    if sample.hand.confidence < min_hand_confidence:
        return False
    if now_ms - sample.timestamp_ms > stale_timeout_ms:
        return False
    return True
```

- Update the only caller in `run_loop` and in `handle_neutral_capture` to drop the `min_pose_visibility=...` kwarg.

- [ ] **Step 4: Update test_main_args.py**

In `playground/teleop2/tests/test_main_args.py`:

- Delete the test `test_invalid_min_pose_visibility_raises_system_exit`.
- Any other test that uses `--min-pose-visibility` should drop it.

- [ ] **Step 5: Update test_main_loop.py**

In `playground/teleop2/tests/test_main_loop.py`:

- Update the `safety_config()` helper to drop `min_pose_visibility`.
- Update all `main.sample_is_usable(...)` calls to drop the `min_pose_visibility=` kwarg.
- Update all `main.handle_neutral_capture(...)` calls to drop the `min_pose_visibility=` kwarg.
- Update the `sample()` helper to build a hand-only TeleopSample (the existing helper constructs ArmSample, but ArmSample is now deleted). New helper:

```python
def sample(timestamp_ms: int = 1000) -> TeleopSample:
    points = [Landmark(0.5, 0.65, 0.0) for _ in range(21)]
    points[5] = Landmark(0.45, 0.55, 0.0)
    points[17] = Landmark(0.55, 0.55, 0.0)
    points[9] = Landmark(0.50, 0.45, 0.0)
    points[4] = Landmark(0.42, 0.40, 0.0)
    points[8] = Landmark(0.58, 0.40, 0.0)
    hand = HandSample(points, handedness="Right", confidence=0.9, timestamp_ms=timestamp_ms)
    return TeleopSample(hand=hand, timestamp_ms=timestamp_ms)
```

Remove the import of `ArmSample, PoseLandmark` from `teleop2.types` in the test imports.

- Replace `test_sample_is_usable_false_when_pose_visibility_below_threshold` with a hand-confidence test:

```python
def test_sample_is_usable_false_when_hand_confidence_below_threshold() -> None:
    base = sample()
    low_confidence_hand = HandSample(
        landmarks=list(base.hand.landmarks),
        handedness=base.hand.handedness,
        confidence=0.2,
        timestamp_ms=base.hand.timestamp_ms,
    )
    s = TeleopSample(hand=low_confidence_hand, timestamp_ms=base.timestamp_ms)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )
```

- Replace `test_sample_is_usable_false_when_arm_missing` with:

```python
def test_sample_is_usable_false_when_hand_missing() -> None:
    s = TeleopSample(hand=None, timestamp_ms=1000)
    assert not main.sample_is_usable(
        s,
        now_ms=1000,
        min_hand_confidence=0.45,
        stale_timeout_ms=200,
    )
```

- Keep the other `sample_is_usable` tests but drop the `min_pose_visibility=` kwarg in each call.

- Update `test_handle_neutral_capture_rejects_unusable_sample` to drop the unusable sample's arm reference — build it as `TeleopSample(hand=None, timestamp_ms=1000)`.

- Update `test_arm_image_landmarks_*` tests — these test a function that won't survive the overlay rewrite in Task 9. Delete them now.

- [ ] **Step 6: Run all tests to verify**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: full suite passes. The TeleopMapper still holds shoulder_pan/lift/elbow_flex at baseline, but everything compiles and tests pass.

- [ ] **Step 7: Commit**

```bash
git add playground/teleop2/teleop2/safety.py playground/teleop2/teleop2/main.py playground/teleop2/tests/
git commit -m "feat(teleop2): drop min_pose_visibility from safety, main, tests"
```

---

## Task 6: Add HandPositionMapper

**Files:**
- Modify: `playground/teleop2/teleop2/pose_mapper.py` (add `HandPositionFeatures`, `extract_hand_position_features`, `HandPositionMapper`, plus 3 new `MappingConfig` fields)
- Modify: `playground/teleop2/tests/test_pose_mapper.py` (append HandPositionMapper tests)

- [ ] **Step 1: Write failing HandPositionMapper tests**

Append to `playground/teleop2/tests/test_pose_mapper.py`:

```python
from teleop2.pose_mapper import (
    HandPositionFeatures,
    HandPositionMapper,
    extract_hand_position_features,
)


def hand_at(*, wrist_xy=(0.50, 0.50), hand_width=0.10, handedness="Right", timestamp_ms=1000):
    """Build a HandSample with hand wrist at wrist_xy and the index_mcp/pinky_mcp separated by hand_width."""
    points = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]
    points[0] = Landmark(wrist_xy[0], wrist_xy[1], 0.0)
    # Spread index_mcp and pinky_mcp horizontally by hand_width centered on the wrist
    points[5] = Landmark(wrist_xy[0] - hand_width / 2, wrist_xy[1] - 0.05, 0.0)
    points[17] = Landmark(wrist_xy[0] + hand_width / 2, wrist_xy[1] - 0.05, 0.0)
    points[9] = Landmark(wrist_xy[0], wrist_xy[1] - 0.05, 0.0)
    points[4] = Landmark(wrist_xy[0] - 0.05, wrist_xy[1] - 0.10, 0.0)
    points[8] = Landmark(wrist_xy[0] + 0.05, wrist_xy[1] - 0.10, 0.0)
    return HandSample(points, handedness=handedness, confidence=0.9, timestamp_ms=timestamp_ms)


def hand_position_mapper(**overrides) -> HandPositionMapper:
    defaults = dict(
        wrist_flex_gain=30.0,
        wrist_roll_gain=60.0,
        gripper_open=80.0,
        gripper_closed=20.0,
        pinch_closed_ratio=0.35,
        pinch_open_ratio=1.40,
        shoulder_pan_gain=60.0,
        shoulder_lift_gain=80.0,
        invert_shoulder_lift=False,
    )
    defaults.update(overrides)
    return HandPositionMapper(MappingConfig(**defaults))


def test_extract_hand_position_features_uses_wrist_xy_and_hand_width() -> None:
    sample = hand_at(wrist_xy=(0.30, 0.60), hand_width=0.12)
    features = extract_hand_position_features(sample, MappingConfig())
    assert features.pan_x == pytest.approx(0.30)
    assert features.hand_size == pytest.approx(0.12, rel=1e-3)


def test_hand_position_mapper_neutral_yields_baseline() -> None:
    mapper = hand_position_mapper()
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(neutral)

    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan, abs=1e-6)
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift, abs=1e-6)


def test_hand_position_mapper_lateral_translation_maps_to_shoulder_pan() -> None:
    mapper = hand_position_mapper(shoulder_pan_gain=100.0)
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    moved = hand_at(wrist_xy=(0.8, 0.5), hand_width=0.10)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(moved)

    # pan_delta = 0.8 - 0.5 = 0.3, gain = 100 → 30
    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan + 30.0)


def test_hand_position_mapper_hand_growth_maps_to_shoulder_lift() -> None:
    mapper = hand_position_mapper(shoulder_lift_gain=100.0)
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    bigger = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.12)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(bigger)

    # size_ratio = (0.12 - 0.10) / 0.10 = 0.20, gain = 100 → 20
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift + 20.0)


def test_hand_position_mapper_invert_lift_flips_sign() -> None:
    mapper = hand_position_mapper(shoulder_lift_gain=100.0, invert_shoulder_lift=True)
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    bigger = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.12)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(bigger)

    # Without invert: +20; with invert: -20
    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift - 20.0)


def test_hand_position_mapper_pan_unchanged_when_only_size_changes() -> None:
    mapper = hand_position_mapper()
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    bigger = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.12)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(bigger)

    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan, abs=1e-6)


def test_hand_position_mapper_lift_unchanged_when_only_pan_changes() -> None:
    mapper = hand_position_mapper()
    neutral = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    moved = hand_at(wrist_xy=(0.8, 0.5), hand_width=0.10)
    mapper.capture_neutral(neutral, baseline_targets())

    targets = mapper.map(moved)

    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift, abs=1e-6)


def test_hand_position_mapper_map_requires_neutral_capture() -> None:
    mapper = hand_position_mapper()
    with pytest.raises(RuntimeError, match="Neutral hand position features have not been captured"):
        mapper.map(hand_at())


def test_hand_position_mapper_capture_neutral_rejects_degenerate_hand_size() -> None:
    mapper = hand_position_mapper()
    # min_hand_width default is 0.03; a width below that should be rejected before any size_ratio math
    degenerate = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.02)
    with pytest.raises(ValueError, match="Hand width is too small"):
        mapper.capture_neutral(degenerate, baseline_targets())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd playground/teleop2 && uv run pytest tests/test_pose_mapper.py -v
```

Expected: New HandPositionMapper tests fail with `ImportError: cannot import name 'HandPositionMapper'`.

- [ ] **Step 3: Add the three new MappingConfig fields**

In `playground/teleop2/teleop2/pose_mapper.py`, extend `MappingConfig` to:

```python
@dataclass(frozen=True)
class MappingConfig:
    wrist_flex_gain: float = 30.0
    wrist_roll_gain: float = 60.0
    gripper_open: float = 80.0
    gripper_closed: float = 20.0
    pinch_closed_ratio: float = 0.35
    pinch_open_ratio: float = 1.40
    min_hand_width: float = 0.03
    mirror_hand: bool = False
    shoulder_pan_gain: float = 60.0
    shoulder_lift_gain: float = 80.0
    invert_shoulder_lift: bool = False

    def __post_init__(self) -> None:
        numeric_fields = (
            "wrist_flex_gain",
            "wrist_roll_gain",
            "gripper_open",
            "gripper_closed",
            "pinch_closed_ratio",
            "pinch_open_ratio",
            "min_hand_width",
            "shoulder_pan_gain",
            "shoulder_lift_gain",
        )
        for field_name in numeric_fields:
            _validate_finite_number(field_name, getattr(self, field_name))

        if self.min_hand_width <= 0.0:
            raise ValueError("min_hand_width must be greater than 0")
        if self.pinch_open_ratio <= self.pinch_closed_ratio:
            raise ValueError("pinch_open_ratio must be greater than pinch_closed_ratio")
```

- [ ] **Step 4: Add HandPositionFeatures, extract_hand_position_features, HandPositionMapper**

Append to `playground/teleop2/teleop2/pose_mapper.py`:

```python
@dataclass(frozen=True)
class HandPositionFeatures:
    pan_x: float
    hand_size: float


def extract_hand_position_features(
    sample: HandSample, config: MappingConfig | None = None
) -> HandPositionFeatures:
    cfg = config or MappingConfig()
    landmarks = sample.landmarks
    if len(landmarks) < 21:
        raise ValueError("HandSample must contain 21 landmarks")

    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    for landmark in (wrist, index_mcp, pinky_mcp):
        _validate_landmark(landmark)

    hand_size = _distance3(index_mcp, pinky_mcp)
    if hand_size < cfg.min_hand_width:
        raise ValueError("Hand width is too small")

    return HandPositionFeatures(pan_x=wrist.x, hand_size=hand_size)


class HandPositionMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self._neutral_features: HandPositionFeatures | None = None
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return self._neutral_features is not None and self._neutral_targets is not None

    def capture_neutral(self, sample: HandSample, robot_targets: RobotTargets) -> None:
        _validate_robot_targets(robot_targets)
        self._neutral_features = extract_hand_position_features(sample, self.config)
        self._neutral_targets = robot_targets

    def map(self, sample: HandSample) -> RobotTargets:
        if self._neutral_features is None or self._neutral_targets is None:
            raise RuntimeError("Neutral hand position features have not been captured")

        features = extract_hand_position_features(sample, self.config)
        pan_delta = features.pan_x - self._neutral_features.pan_x
        size_ratio = (features.hand_size - self._neutral_features.hand_size) / self._neutral_features.hand_size
        if self.config.invert_shoulder_lift:
            size_ratio = -size_ratio

        return RobotTargets(
            shoulder_pan=self._neutral_targets.shoulder_pan
            + pan_delta * self.config.shoulder_pan_gain,
            shoulder_lift=self._neutral_targets.shoulder_lift
            + size_ratio * self.config.shoulder_lift_gain,
            elbow_flex=self._neutral_targets.elbow_flex,
            wrist_flex=self._neutral_targets.wrist_flex,
            wrist_roll=self._neutral_targets.wrist_roll,
            gripper=self._neutral_targets.gripper,
        )
```

- [ ] **Step 5: Run pose_mapper tests to confirm they pass**

```bash
cd playground/teleop2 && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All pose_mapper tests pass (old WristMapper + new HandPositionMapper).

- [ ] **Step 6: Commit**

```bash
git add playground/teleop2/teleop2/pose_mapper.py playground/teleop2/tests/test_pose_mapper.py
git commit -m "feat(teleop2): add HandPositionMapper for shoulder_pan/lift from hand position+size"
```

---

## Task 7: Wire HandPositionMapper into TeleopMapper

**Files:**
- Modify: `playground/teleop2/teleop2/pose_mapper.py` (replace the stub `TeleopMapper`)
- Modify: `playground/teleop2/tests/test_pose_mapper.py` (extend TeleopMapper tests)

- [ ] **Step 1: Write failing tests for the wired TeleopMapper**

Append to `playground/teleop2/tests/test_pose_mapper.py`:

```python
def test_teleop_mapper_applies_shoulder_pan_from_hand_position() -> None:
    mapper = teleop_mapper(shoulder_pan_gain=100.0)
    neutral_hand = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    moved_hand = hand_at(wrist_xy=(0.8, 0.5), hand_width=0.10)
    mapper.capture_neutral(
        TeleopSample(hand=neutral_hand, timestamp_ms=1000), baseline_targets()
    )

    targets = mapper.map(TeleopSample(hand=moved_hand, timestamp_ms=1000))

    assert targets.shoulder_pan == pytest.approx(baseline_targets().shoulder_pan + 30.0)


def test_teleop_mapper_applies_shoulder_lift_from_hand_size() -> None:
    mapper = teleop_mapper(shoulder_lift_gain=100.0)
    neutral_hand = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    bigger_hand = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.12)
    mapper.capture_neutral(
        TeleopSample(hand=neutral_hand, timestamp_ms=1000), baseline_targets()
    )

    targets = mapper.map(TeleopSample(hand=bigger_hand, timestamp_ms=1000))

    assert targets.shoulder_lift == pytest.approx(baseline_targets().shoulder_lift + 20.0)


def test_teleop_mapper_always_holds_elbow_flex_at_baseline() -> None:
    mapper = teleop_mapper(shoulder_pan_gain=100.0, shoulder_lift_gain=100.0)
    neutral_hand = hand_at(wrist_xy=(0.5, 0.5), hand_width=0.10)
    moved_hand = hand_at(wrist_xy=(0.8, 0.5), hand_width=0.12)  # both pan and lift change
    mapper.capture_neutral(
        TeleopSample(hand=neutral_hand, timestamp_ms=1000), baseline_targets()
    )

    targets = mapper.map(TeleopSample(hand=moved_hand, timestamp_ms=1000))

    assert targets.elbow_flex == pytest.approx(baseline_targets().elbow_flex)


def test_teleop_mapper_neutral_ready_requires_both_wrist_and_position() -> None:
    mapper = teleop_mapper()
    assert mapper.neutral_ready is False
    mapper.capture_neutral(teleop_sample(), baseline_targets())
    assert mapper.neutral_ready is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd playground/teleop2 && uv run pytest tests/test_pose_mapper.py -v
```

Expected: The four new tests fail because the stub `TeleopMapper` still returns baseline for shoulder_pan/lift.

- [ ] **Step 3: Replace the TeleopMapper stub**

In `playground/teleop2/teleop2/pose_mapper.py`, replace the entire `TeleopMapper` class with:

```python
class TeleopMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.wrist = WristMapper(config)
        self.position = HandPositionMapper(config)
        self._neutral_targets: RobotTargets | None = None

    @property
    def neutral_ready(self) -> bool:
        return (
            self.wrist.neutral_ready
            and self.position.neutral_ready
            and self._neutral_targets is not None
        )

    def capture_neutral(self, sample: TeleopSample, robot_targets: RobotTargets) -> None:
        if sample.hand is None:
            raise ValueError("TeleopMapper neutral capture requires a hand sample")
        self.wrist.capture_neutral(sample.hand, robot_targets)
        self.position.capture_neutral(sample.hand, robot_targets)
        self._neutral_targets = robot_targets

    def map(self, sample: TeleopSample) -> RobotTargets:
        if sample.hand is None:
            raise ValueError("TeleopMapper map requires a hand sample")
        if self._neutral_targets is None:
            raise RuntimeError("Neutral TeleopMapper sample has not been captured")

        wrist_targets = self.wrist.map(sample.hand)
        position_targets = self.position.map(sample.hand)
        return RobotTargets(
            shoulder_pan=position_targets.shoulder_pan,
            shoulder_lift=position_targets.shoulder_lift,
            elbow_flex=self._neutral_targets.elbow_flex,
            wrist_flex=wrist_targets.wrist_flex,
            wrist_roll=wrist_targets.wrist_roll,
            gripper=wrist_targets.gripper,
        )
```

- [ ] **Step 4: Run pose_mapper tests to confirm they pass**

```bash
cd playground/teleop2 && uv run pytest tests/test_pose_mapper.py -v
```

Expected: All pose_mapper tests pass (WristMapper, HandPositionMapper, and TeleopMapper).

- [ ] **Step 5: Run the full suite — main.py tests should still pass too**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add playground/teleop2/teleop2/pose_mapper.py playground/teleop2/tests/test_pose_mapper.py
git commit -m "feat(teleop2): wire HandPositionMapper into TeleopMapper for live shoulder control"
```

---

## Task 8: Add --invert-shoulder-lift and adjust gain defaults in main.py

**Files:**
- Modify: `playground/teleop2/main.py`
- Modify: `playground/teleop2/tests/test_main_args.py`

- [ ] **Step 1: Write failing tests for the new flag**

Append to `playground/teleop2/tests/test_main_args.py`:

```python
def test_invert_shoulder_lift_defaults_to_false() -> None:
    args = parsed()
    main.apply_camera_defaults(args)
    main.validate_args(args)
    assert args.invert_shoulder_lift is False


def test_invert_shoulder_lift_flag_sets_true() -> None:
    args = parsed("--invert-shoulder-lift")
    main.apply_camera_defaults(args)
    main.validate_args(args)
    assert args.invert_shoulder_lift is True


def test_default_shoulder_lift_gain_is_eighty() -> None:
    args = parsed()
    assert args.shoulder_lift_gain == pytest.approx(80.0)
```

Add `import pytest` at the top of the file if it isn't already there.

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd playground/teleop2 && uv run pytest tests/test_main_args.py -v
```

Expected: the three new tests fail with `AttributeError: 'Namespace' object has no attribute 'invert_shoulder_lift'`.

- [ ] **Step 3: Add the flag and adjust the default**

In `playground/teleop2/main.py`, `parse_args`:

- Change the existing `parser.add_argument("--shoulder-lift-gain", type=float, default=20.0)` default to `80.0`.
- Add: `parser.add_argument("--invert-shoulder-lift", action="store_true")`.
- Drop the `--arm`, `--mirror-hand` flags **NO — actually keep `--mirror-hand`** (it's still useful for egocentric POV) and drop `--arm` (we no longer pick between left/right arm because we don't track the arm). Actually re-evaluate: `--mirror-hand` still applies because the hand orientation in egocentric is still palm-away. Keep it. Drop `--arm` only.
- Actually leave both `--arm` and `--mirror-hand` alone for now — they're harmless. `--arm` is dead code but removing it requires more test updates. We'll let it ride.

In `make_mapping_config`:

- Add `invert_shoulder_lift=args.invert_shoulder_lift` to the `MappingConfig(...)` call.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd playground/teleop2 && uv run pytest tests/test_main_args.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run the full suite**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add playground/teleop2/main.py playground/teleop2/tests/test_main_args.py
git commit -m "feat(teleop2): add --invert-shoulder-lift flag and bump --shoulder-lift-gain default"
```

---

## Task 9: Overlay — drop arm skeleton, add neutral crosshair and size circle

**Files:**
- Modify: `playground/teleop2/teleop2/tracker.py` (rewrite `draw_overlay` to drop arm code and add neutral references)
- Modify: `playground/teleop2/tests/test_tracker.py` (rewrite overlay tests)
- Modify: `playground/teleop2/main.py` (pass neutral position to `draw_overlay`, drop `_arm_image_landmarks`)
- Modify: `playground/teleop2/tests/test_main_loop.py` (any remaining overlay-related test cleanups)

- [ ] **Step 1: Write failing test for the new overlay shape**

In `playground/teleop2/tests/test_tracker.py`, delete the existing overlay tests (`test_draw_overlay_does_not_modify_when_samples_are_none`, `test_draw_overlay_draws_pose_skeleton_lines_when_arm_present`, `test_draw_overlay_draws_hand_landmarks_when_hand_present`) and replace with:

```python
import numpy as np

from teleop2.tracker import draw_overlay
from teleop2.types import HandSample, Landmark


def test_draw_overlay_does_not_modify_when_no_hand_and_no_status() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    snapshot = frame.copy()
    draw_overlay(
        frame,
        hand=None,
        status_lines=[],
        image_size=(320, 240),
        neutral_reference=None,
    )
    assert np.array_equal(frame, snapshot)


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
        hand=hand,
        status_lines=[],
        image_size=(320, 240),
        neutral_reference=None,
    )
    assert frame.sum() > 0


def test_draw_overlay_draws_neutral_crosshair_and_size_circle_when_reference_given() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_overlay(
        frame,
        hand=None,
        status_lines=[],
        image_size=(320, 240),
        neutral_reference={"pan_x": 0.5, "wrist_y": 0.5, "hand_size": 0.1},
    )
    assert frame.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd playground/teleop2 && uv run pytest tests/test_tracker.py -v
```

Expected: the new overlay tests fail because `draw_overlay`'s signature still has `arm` and `arm_image_landmarks`.

- [ ] **Step 3: Rewrite draw_overlay in tracker.py**

In `playground/teleop2/teleop2/tracker.py`, replace the entire `draw_overlay` function with:

```python
def draw_overlay(
    frame,
    *,
    hand: HandSample | None,
    status_lines: list[str],
    image_size: tuple[int, int],
    neutral_reference: dict | None = None,
) -> None:
    width, height = image_size

    if neutral_reference is not None:
        pan_x = neutral_reference["pan_x"]
        wrist_y = neutral_reference["wrist_y"]
        hand_size = neutral_reference["hand_size"]
        cx = max(0, min(width - 1, int(pan_x * width)))
        cy = max(0, min(height - 1, int(wrist_y * height)))
        radius = max(2, int(hand_size * width / 2))
        # Vertical crosshair line at the neutral pan_x (helps see lateral drift)
        cv2.line(frame, (cx, 0), (cx, height - 1), (0, 165, 255), 1, cv2.LINE_AA)
        # Reference circle sized to the neutral hand size (helps see depth drift)
        cv2.circle(frame, (cx, cy), radius, (0, 165, 255), 1, cv2.LINE_AA)

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

- [ ] **Step 4: Update main.py to call the new overlay signature and pass a neutral reference**

In `playground/teleop2/main.py`:

- Delete the `_arm_image_landmarks` helper function entirely.
- In `run_loop`, replace the `draw_overlay(...)` call with:

```python
neutral_reference = None
if mapper.neutral_ready and mapper.position._neutral_features is not None:
    # Surface the captured neutral so the overlay can show the reference crosshair + size circle.
    nf = mapper.position._neutral_features
    # Find the neutral wrist's y from the wrist mapper's captured neutral hand sample.
    # We didn't store it; fall back to the centre of the frame.
    neutral_reference = {
        "pan_x": nf.pan_x,
        "wrist_y": 0.5,
        "hand_size": nf.hand_size,
    }

draw_overlay(
    frame,
    hand=teleop_sample.hand,
    status_lines=status_lines,
    image_size=(args.width, args.height),
    neutral_reference=neutral_reference,
)
```

Note: we're using a private attribute (`_neutral_features`) here. For a cleaner design, add a `neutral_reference` property to `HandPositionMapper`. Do that now: in `pose_mapper.py`, append to `HandPositionMapper`:

```python
    @property
    def neutral_reference(self) -> dict | None:
        if self._neutral_features is None:
            return None
        return {"pan_x": self._neutral_features.pan_x, "hand_size": self._neutral_features.hand_size}
```

And change main.py's neutral_reference construction to:

```python
neutral_reference = None
if mapper.position.neutral_reference is not None:
    nr = mapper.position.neutral_reference
    neutral_reference = {"pan_x": nr["pan_x"], "wrist_y": 0.5, "hand_size": nr["hand_size"]}
```

- Update `build_status_lines` to drop any pose-related fields. The function should now produce:

```python
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
    hand_state = "none" if sample.hand is None else f"{sample.hand.handedness} {sample.hand.confidence:.2f}"
    clamp_text = ",".join(result.clamped_keys) if result.clamped_keys else "none"
    targets = result.targets
    line1 = (
        f"{robot_state} | {fps_display:4.1f} FPS | sync={'on' if sync_enabled else 'off'} | "
        f"neutral={'yes' if neutral_ready else 'no'} | reason={result.reason.value} | clamp={clamp_text}"
    )
    line2 = f"hand={hand_state}"
    line3 = (
        f"pan={targets.shoulder_pan:.1f} lift={targets.shoulder_lift:.1f} "
        f"elb={targets.elbow_flex:.1f} wf={targets.wrist_flex:.1f} "
        f"wr={targets.wrist_roll:.1f} grip={targets.gripper:.1f}"
    )
    lines = [line1, line2, line3]
    if notice:
        lines.append(f"notice: {notice}")
    return lines
```

- [ ] **Step 5: Run all tests**

```bash
cd playground/teleop2 && uv run pytest -v
```

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add playground/teleop2/teleop2/pose_mapper.py playground/teleop2/teleop2/tracker.py playground/teleop2/main.py playground/teleop2/tests/test_tracker.py
git commit -m "feat(teleop2): rewrite overlay for hand-only with neutral crosshair + size circle"
```

---

## Task 10: End-to-end `--check` smoke test on real binary

**Files:**
- None expected to modify (verification task).

- [ ] **Step 1: Run --check and confirm it succeeds**

```bash
cd playground/teleop2 && uv run python main.py --check
```

Expected: downloads (first run) or finds (subsequent run) the Hand Landmarker model, prints `Hand model ready: ...`, exits 0.

If the script tries to download or reference a Pose model, that means Task 2 missed a Pose-related reference. Find it, delete it, re-run.

- [ ] **Step 2: If it works first try, no commit needed**

- [ ] **Step 3: If a bug surfaced, fix with a minimal TDD step and commit**

```bash
git add playground/teleop2/<file>
git commit -m "fix(teleop2): <what>"
```

---

## Task 11: README rewrite

**Files:**
- Modify: `playground/teleop2/README.md`

- [ ] **Step 1: Rewrite the README**

Replace `playground/teleop2/README.md` contents with:

````markdown
# Egocentric SO101 5-DOF Hand-Only Teleop

Hand-only egocentric webcam teleoperation prototype for the SO101 follower.
A slanted top-down webcam looks at the user's hand on or above a desk; MediaPipe
Hand Landmarker drives five follower joints. `elbow_flex` is held at the
operator's starting position for the entire session.

This is the hand-only companion to `playground/teleop/`, which uses an
egocentric chest-mounted camera with both MediaPipe Pose and Hand to drive all
six joints. Trade-off: teleop2 is much more robust (hand tracking rarely fails)
but gives up `elbow_flex` control.

## Setup

```bash
uv sync
```

## Check

Validates the model download and (optionally) the robot config without opening
the camera:

```bash
uv run python main.py --check
```

## Dry Run

```bash
uv run python main.py --camera-index 0 --fps 15
```

Controls:

- `n`: capture neutral hand pose. Hand should be tracked with high confidence
  in the centre of the frame at a comfortable distance from the camera.
- `space`: toggle real-time sync.
- `q` or `Esc`: exit.

Before neutral capture, no targets are emitted.

## Mapping

- `shoulder_pan.pos` — hand wrist landmark's x position in the frame, delta
  from neutral, scaled by `--shoulder-pan-gain`.
- `shoulder_lift.pos` — hand's apparent size (distance between index and pinky
  MCPs) relative to neutral, scaled by `--shoulder-lift-gain`. Bigger hand =
  arm is lifted (because closer to the slanted top-down camera).
- `wrist_flex.pos` — hand flex relative to neutral (same math as exocentric).
- `wrist_roll.pos` — palm tilt relative to neutral.
- `gripper.pos` — continuous thumb-index pinch openness (absolute, not
  neutral-relative).
- `elbow_flex.pos` — held at the operator's startup position.

Use `--invert-shoulder-lift` if your robot's calibration has the opposite
sign convention for shoulder lift.

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
  --wrist-flex-limit 10 \
  --wrist-roll-limit 20 \
  --deadman-key x
```

## Camera Placement

A webcam slanted top-down onto the desk where you'll work. Your hand should be
in the centre of the frame at neutral, with comfortable room to translate
left/right and toward/away-from the camera. No body, shoulder, or elbow needs
to be in frame.

When you capture neutral, the overlay draws an orange vertical line at your
neutral hand x position and an orange circle sized to your neutral hand. As you
teleoperate, you can see at a glance whether your hand has drifted from neutral
(line position) and whether it's at the right depth (circle size vs current
hand size).

## Model Cache

The Hand Landmarker model is downloaded on first use into `models/` and ignored
by git.

## Safety Notes

This commands five SO101 follower joints (`elbow_flex` is held). Always:

- Start with `--deadman-key x` and conservative limits.
- Keep the robot's workspace clear; the shoulder has a much larger swept volume
  than the wrist alone.
- The script freezes command output when tracking is missing, stale, below
  confidence, paused, the deadman key is inactive, or neutral has not been
  captured.
- A backend send failure disables sync and locks command output off until
  restart.
````

- [ ] **Step 2: Commit**

```bash
git add playground/teleop2/README.md
git commit -m "docs(teleop2): rewrite README for hand-only 5-DOF variant"
```

---

## Self-Review

**1. Spec coverage:** Cross-checking the spec sections against tasks:

- Directory structure → Task 1.
- Hand-only types → Tasks 3 + 5 (TeleopSample lost arm field; sample_is_usable simplified).
- Wrist mapping (reused) → Task 1 (lifted) and Task 4 (cleaned up MappingConfig).
- Hand position mapping math → Task 6.
- TeleopMapper wiring + elbow_flex held → Task 7.
- Safety dropping `min_pose_visibility` → Task 5.
- Hand-only tracker → Task 2.
- Overlay (drop arm skeleton, add neutral crosshair + size circle) → Task 9.
- `--invert-shoulder-lift` flag + gain defaults → Task 8.
- main.py loop simplification → Tasks 5 + 9.
- End-to-end --check → Task 10.
- README → Task 11.

All spec sections covered.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", "similar to Task N" without repeated code. Each step has actual code or precise instructions.

**3. Type consistency:** `MappingConfig` field names are consistent across Tasks 4, 6, 8. `TeleopSample` is hand-only after Task 3 and stays that way. `HandPositionFeatures(pan_x, hand_size)` is referenced consistently in Tasks 6 + 7 + 9 (overlay reads via `neutral_reference` dict with matching keys). `draw_overlay` signature in Task 9 matches the call site in Task 9's main.py update. `SafetyConfig` after Task 5 stays at 5 fields throughout subsequent tasks.

**One known oddity:** Task 9 stashes `wrist_y: 0.5` (centre of frame) in the `neutral_reference` because we don't actually capture neutral wrist_y separately — the overlay's crosshair is purely vertical at the neutral pan_x, and the circle is centred at (pan_x, frame_centre_y). The vertical line is the meaningful reference for lateral drift; the y-centring of the circle is just a visual anchor. If the implementer wants to improve this, they can extend `HandPositionMapper` to also store the neutral wrist y — out of scope for this plan.
