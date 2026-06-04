# Egocentric SO101 6-DOF Teleoperation Design

Date: 2026-05-24

## Goal

Build a new playground prototype at `playground/teleop/` that explores real-time SO101 follower teleoperation from a single **egocentric** RGB webcam — that is, a camera mounted on the user's chest (or similar) that looks forward-and-down at the user's own arm and hand.

Where the existing `playground/mediapipe_so101/` exocentric prototype controls only the wrist and gripper (3 of 6 joints) because a front-facing camera cannot reliably see the user's elbow or shoulder, the egocentric viewpoint observes the forearm, elbow, and often part of the shoulder. This unlocks **full 6-DOF arm control**: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`.

The prototype defaults to a dry run and only commands physical robot motion when launched with an explicit `--enable-robot` flag.

## Non-Goals

This prototype will not implement inverse kinematics, dataset recording, learned policies, bimanual control, multi-camera fusion, depth sensors, IMU fusion, hot-reloadable mapping configs, or a web UI. It will not modify the existing `playground/mediapipe_so101/` exocentric playground — that prototype keeps its narrower control surface and remains the simpler reference implementation.

## Runtime Behavior

The script starts in a paused state. It opens the camera, runs MediaPipe Pose Landmarker and Hand Landmarker concurrently, and displays the detected arm and hand overlay. No robot commands are emitted until a neutral pose has been captured.

The user holds their arm in a comfortable resting posture (forearm in view, hand relaxed open) and presses `n` to capture neutral. Neutral stores the current arm and hand baseline so that all subsequent arm and wrist targets are reported as deltas from neutral. The gripper command is not neutral-relative; it remains an absolute mapping from current pinch openness to the configured open/closed targets, mirroring the exocentric playground.

The user presses `space` to toggle sync. In dry-run mode, sync updates target values shown on the overlay. In robot mode, sync sends filtered targets to the SO101 follower at the configured control rate. An optional `--deadman-key` mode requires recent key activity before commands flow, in addition to the sync toggle.

The `q` or `Esc` key exits cleanly. Exit, keyboard interrupt, camera failure, or robot connection failure must leave the robot disconnected through LeRobot's normal cleanup path. A backend send failure locks sync off until restart, the same as the exocentric playground.

## Architecture

The prototype mirrors the four-stage `tracker → mapper → safety → backend` pipeline of `playground/mediapipe_so101/`. It is organized as plain Python modules under `playground/teleop/`:

- `README.md`: setup, dry-run command, robot command, key controls, safety notes, tuning tips, egocentric-specific guidance (mount placement, lighting).
- `pyproject.toml`: dependencies for MediaPipe, OpenCV, and the local LeRobot-compatible runtime. Same dependency surface as `playground/mediapipe_so101/`.
- `main.py`: command-line parsing and the camera/control loop.
- `teleop/types.py`: `Landmark`, `HandSample`, `ArmSample`, `TeleopSample`, `RobotTargets` (now carrying all 6 joints), `FilterResult`, `FreezeReason`.
- `teleop/tracker.py`: MediaPipe Pose and Hand model download, both landmarker creations, latest-result buffering for each, per-frame fusion, camera helpers, and overlay drawing.
- `teleop/pose_mapper.py`: `ArmMapper` (arm landmarks → 3 arm-joint targets), `WristMapper` (hand landmarks → 3 wrist+gripper targets, math lifted from the exocentric playground), and `TeleopMapper` (combines both with a shared neutral-capture interface).
- `teleop/safety.py`: 6-joint extension of the exocentric `TargetFilter`, plus the new visibility-based freeze rule for pose landmarks.
- `teleop/robot_backend.py`: `DryRunBackend` and `SO101Backend` for the full 6-joint action surface.

`pose_mapper.py` and `safety.py` must remain usable without a camera or hardware.

**Reuse vs duplication.** Each playground is its own uv project. The wrist/gripper mapping math is lifted from `mediapipe_so101/pose_mapper.py` rather than imported — this preserves project isolation and lets the two playgrounds evolve independently.

## Perception

Two MediaPipe Tasks LIVE_STREAM landmarkers run concurrently against each frame:

- **Pose Landmarker** produces `pose_landmarks` (33 landmarks, normalized image coords) and `pose_world_landmarks` (33 landmarks in meters, hip-centered, right-handed). The arm features use only the world landmarks; the image landmarks are used solely to match the Pose wrist to a Hand result.
- **Hand Landmarker** produces 21 hand landmarks in normalized image coords for each detected hand, exactly as in the exocentric playground.

Each landmarker has its own result callback updating a thread-safe `LatestPoseResult` / `LatestHandResult`. The control loop reads the latest from both, picks the Pose result for the configured tracked arm (`--arm right` by default), then picks the Hand result whose wrist landmark is closest in image-plane distance to the Pose wrist. Handedness labels are not used for matching — in egocentric POV the back of the hand faces the camera and labels are unreliable.

## Fusion

The fusion step produces a `TeleopSample`:

```
TeleopSample {
    arm: ArmSample(shoulder, elbow, wrist, in pose_world coords, with per-landmark visibility),
    hand: HandSample(21 landmarks, normalized image coords, with confidence and handedness),
    timestamp_ms,
}
```

If either side is missing or stale, the corresponding field is `None` and the safety filter freezes for that reason (see Safety).

## Mapping

The arm features are computed in a shoulder-local frame so whole-body translation does not drift the targets. Translate so the shoulder is the origin, define `upper_arm = elbow - shoulder` and `forearm = wrist - elbow`, then compute three independent features.

`elbow_flex` is the angle between `upper_arm` and `forearm`:

```
elbow_angle = acos( (upper_arm · forearm) / (|upper_arm| * |forearm|) )
```

Range: ~0 when the arm is fully straight (vectors aligned) to ~π when the arm is fully folded against itself (vectors anti-parallel). This is the geometric angle between the two segments, not the anatomical "elbow flexion" angle.

`shoulder_lift` is the pitch of `upper_arm` relative to the body's vertical axis (Pose convention: +y is down):

```
horizontal = sqrt(upper_arm.x² + upper_arm.z²)
shoulder_lift = atan2(horizontal, upper_arm.y)
```

Range: ~0 when arm hangs straight down (upper_arm.y > 0), ~π/2 when arm is horizontal, ~π when arm points straight up.

`shoulder_pan` is the yaw of `upper_arm` around the body's vertical axis:

```
shoulder_pan = atan2(upper_arm.x, upper_arm.z)
```

All three arm features are reported as a delta from the neutral capture, scaled by per-joint gain CLI flags (`--shoulder-pan-gain`, `--shoulder-lift-gain`, `--elbow-flex-gain`). Exact angle formulas can be adjusted during implementation, but they must be deterministic and documented in code.

The wrist and gripper features are lifted directly from `mediapipe_so101/pose_mapper.py`:

- `wrist_roll` from `atan2(index_mcp.y - pinky_mcp.y, pinky_mcp.x - index_mcp.x)`, delta from neutral, scaled by `--wrist-roll-gain`.
- `wrist_flex` from `(middle_mcp.y - wrist.y) / hand_width`, delta from neutral, scaled by `--wrist-flex-gain`.
- `gripper` from the thumb-tip to index-tip pinch distance normalized by hand width, mapped continuously from `--pinch-closed-ratio` / `--pinch-open-ratio` to `--gripper-closed` / `--gripper-open`. This mapping is absolute, not neutral-relative.

**Egocentric handedness caveat.** In egocentric POV the back of the hand faces the camera, so the MediaPipe Hand Landmarker may flip its handedness label or invert the sign of the roll feature relative to the exocentric playground. A `--mirror-hand {auto,on,off}` CLI flag controls this; `auto` infers from the Pose handedness convention plus the `--arm` flag, `on` and `off` force the sign.

The combined mapper emits a `RobotTargets` carrying all six joints. Gains, clamps, smoothing coefficient, FPS, stale timeout, arm selection, visibility threshold, and gripper open/closed calibration must all be CLI flags.

## Robot Backend

Dry-run mode is the default and never opens the follower serial port. It tracks a baseline `RobotTargets` for all six joints (the gripper baseline matches `--gripper-open`, the rest default to `0.0`) so the mapper has a consistent neutral to layer deltas onto.

Robot mode requires `--enable-robot`, a follower port, a robot id, and a calibration directory or existing calibration path compatible with the local LeRobot setup. On startup, the backend connects to `SO101Follower`, reads the current observation, and records startup positions for all six joints as the baseline. Subsequent actions include all six action keys; the held-fixed concept from the exocentric playground does not apply here because all six joints are commanded.

The backend interface (`baseline_targets` property, `connect`, `send(RobotTargets)`, `disconnect`) is unchanged from the exocentric playground.

## Safety

The exocentric playground's `TargetFilter` extends to six joints. Per-joint configuration:

- **Limits** are `(baseline - limit, baseline + limit)` for the five delta-driven joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`), each with its own `--*-limit` CLI flag. The gripper limit remains absolute: `(--gripper-min, --gripper-max)`.
- **Max delta** is a single `--max-delta` per-tick rate limit applied uniformly to all six joints. The default is conservative (2.0 in the LeRobot normalized space, half of the exocentric default) because three new big-swing joints are now in play.
- **Smoothing** and **stale timeout** are unchanged from the exocentric playground.

The conservative defaults for the new arm-joint limits are `--shoulder-pan-limit 20`, `--shoulder-lift-limit 20`, `--elbow-flex-limit 25`. These can be increased once the operator has observed the system behave correctly.

### Visibility-based freeze

The genuinely new safety rule. MediaPipe Pose's per-landmark `visibility` score is the signal we use to detect that the arm has left the camera frame. If any of `shoulder`, `elbow`, or `wrist` visibility drops below `--min-pose-visibility` (default 0.6), the filter freezes with `FreezeReason.TRACKING_LOST` and holds targets at the last sent values. This is fail-safe by design: holding stale arm targets when a landmark dropped out would be more dangerous than freezing.

The existing freeze reasons (`PAUSED`, `NEUTRAL_MISSING`, `TRACKING_LOST`, `STALE_RESULT`) and the existing failure-mode handling (backend send failure → sync locked off) carry over unchanged.

## Data Flow

Each frame follows this path:

1. The camera frame is captured with OpenCV and optionally horizontally mirrored.
2. The frame is dispatched to both the Pose Landmarker and the Hand Landmarker via `detect_async`.
3. Both result callbacks update their respective `Latest*Result` buffers.
4. The control loop reads the latest Pose result and selects the configured tracked arm.
5. The control loop reads the latest Hand result and selects the Hand whose wrist is closest in image plane to the Pose wrist.
6. The fused `TeleopSample` is checked for usability: per-landmark Pose visibility above threshold, Hand confidence above threshold, both timestamps within `--stale-timeout-ms`.
7. If neutral has not been captured, status is updated but no target is emitted.
8. After neutral capture, `ArmMapper` produces arm-joint targets from `pose_world_landmarks` and `WristMapper` produces wrist and gripper targets from hand landmarks. The combined `RobotTargets` is passed to the safety filter.
9. The safety filter applies smoothing, clamps, per-frame movement limits, and the freeze rules above.
10. The dry-run backend displays the result on the overlay, or the robot backend sends the filtered action to `SO101Follower.send_action`.

## User Interface

The control surface is identical to the exocentric playground:

- `n` capture neutral. Neutral capture now requires both the Pose arm and the Hand sample to pass their respective usability checks simultaneously; otherwise the capture is rejected with a notice.
- `space` toggle sync on or off.
- `q` or `Esc` exit cleanly.
- `--deadman-key x` and `--deadman-grace-ms 175` for an optional dead-man activation pattern.

The overlay extends the exocentric playground's status line to show all six target values, the active freeze reason, and the per-frame clamp keys. It draws the in-frame portion of the Pose arm skeleton (shoulder → elbow → wrist) in one color and the Hand 21-landmark skeleton in another. The visibility score of each arm landmark is rendered as a small number next to it — invaluable for tuning `--min-pose-visibility`.

## Testing

Tests follow the exocentric playground's pattern: pure-function unit tests with synthetic inputs, no live camera or robot.

`test_pose_mapper.py` covers:

- `ArmMapper` with synthetic 3-point arm poses for canonical configurations (arm hanging down, arm horizontal forward, arm horizontal sideways, elbow bent 90°), verifying expected joint angles within tolerance.
- `ArmMapper` delta-from-neutral correctness: capture neutral at pose A, map pose B, verify the expected delta.
- `ArmMapper` invariance to whole-body translation: translate all three landmarks by a constant offset, verify targets are unchanged.
- `WristMapper` tests lifted from the exocentric `test_pose_mapper.py`.
- `TeleopMapper` combine tests: `capture_neutral` requires both halves to be usable; combined `RobotTargets` carries through correctly.

`test_safety.py` covers the 6-joint extension: each joint clamps independently to its limit, `max_delta` rate-limits each joint independently, each `FreezeReason` fires correctly. Specifically, build a `TeleopSample` with shoulder visibility below threshold and assert `FreezeReason.TRACKING_LOST` with targets held at last-good values.

`test_tracker.py` covers `default_model_path` and `ensure_model` for both Pose and Hand models, the fusion logic (synthetic Pose + multiple Hand results, verify the correct Hand is matched to the chosen Pose wrist by image-plane distance), and thread-safe latest-result updates under simulated async callbacks.

`test_robot_backend.py` verifies that `DryRunBackend` returns a 6-DOF baseline and accepts 6-DOF targets, and that `SO101Backend` wires the right configuration and sends the right 6-key action dict, using the same mocking pattern as the exocentric tests.

Real MediaPipe inference, real robot motion, and live FPS/latency measurement remain manual verification, documented in the README.

## Manual Verification Plan

1. `uv run python main.py --check` validates the model files and configuration without opening the camera.
2. Dry-run with overlay: `uv run python main.py` captures neutral and verifies that each joint target moves sensibly when the arm and hand are moved through canonical poses. Verify the visibility scores behave as expected as the arm leaves the frame.
3. Robot mode with conservative defaults (`--max-delta 2.0`, narrow per-joint limits), then progressively widen after observing safe behavior. Always start with `--deadman-key x` enabled for the first physical run.
