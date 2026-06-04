# MediaPipe SO101 Wrist Teleoperation Design

Date: 2026-05-23

## Goal

Build a third playground prototype at `playground/mediapipe_so101/` that explores real-time SO101 follower wrist and gripper teleoperation from MediaPipe hand pose estimation.

The prototype should combine the existing `playground/mediapipe` webcam hand-tracking demo with the local LeRobot SO101 follower setup in `playground/so101`. It should default to a dry run, and it should only command physical robot motion when launched with an explicit `--enable-robot` flag.

The first version controls only:

- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

All shoulder, elbow, and base joints stay at their startup positions.

## Non-Goals

This prototype will not implement full-arm teleoperation, inverse kinematics, dataset recording, policy inference, bimanual control, or a web UI. It will not modify the existing `playground/mediapipe` or `playground/so101` demos except where documentation needs to reference the new prototype.

## Runtime Behavior

The script starts in a paused state. It opens the camera, runs MediaPipe hand tracking, and displays or logs the detected hand state. No robot commands are emitted until a neutral pose has been captured.

The user presses `n` to capture neutral. Neutral stores the current hand-pose baseline for wrist deltas and, in robot mode, the current robot wrist/gripper targets for initial filter state and held-joint startup posture. The gripper command is not neutral-relative: it remains an absolute mapping from current pinch openness to the configured gripper closed/open targets.

The user presses `space` to toggle sync on or off. In dry-run mode, sync means target values are updated in the display or terminal. In robot mode, sync means filtered targets are sent to the SO101 follower at the configured control rate.

An optional `--deadman-key` mode requires recent key activity before commands flow. This is in addition to the sync toggle. The default interaction remains toggle-based because it is easier for one-person testing, while the deadman mode is available for more cautious physical runs.

The `q` key exits cleanly. Exit, keyboard interrupt, camera failure, or robot connection failure must leave the robot disconnected through LeRobot's normal cleanup path.

## Architecture

The prototype is organized as plain Python modules under `playground/mediapipe_so101/`:

- `README.md`: setup, dry-run command, robot command, key controls, safety notes, and tuning tips.
- `pyproject.toml`: dependencies for MediaPipe, OpenCV, and the local LeRobot-compatible runtime.
- `main.py`: command-line parsing and the camera/control loop.
- `hand_tracker.py`: MediaPipe model download, landmarker creation, latest-result buffering, and camera helpers adapted from `playground/mediapipe`.
- `pose_mapper.py`: landmark feature extraction, neutral capture, and hand pose to SO101 target mapping.
- `robot_backend.py`: dry-run backend and SO101 follower backend behind the same small interface.
- `safety.py`: target smoothing, hard clamps, per-frame rate limits, stale-result handling, and pause/freeze state.

The modules should stay small and testable. `pose_mapper.py` and `safety.py` must be usable without opening a camera or connecting to hardware.

## Data Flow

Each frame follows this path:

1. The camera frame is captured with OpenCV.
2. MediaPipe Hand Landmarker processes the frame in live-stream mode.
3. The latest result is selected if it is recent enough and has adequate confidence.
4. One controlling hand is chosen, defaulting to the highest-confidence hand.
5. If neutral has not been captured, status is updated but no target is emitted.
6. After neutral capture, wrist hand features are computed relative to the neutral hand pose.
7. Relative wrist features and absolute pinch openness are mapped to `wrist_flex`, `wrist_roll`, and `gripper` targets.
8. The safety filter applies smoothing, clamps, and per-frame movement limits.
9. The dry-run backend displays targets, or the robot backend sends the filtered action to `SO101Follower.send_action`.

## Mapping

The mapping should be intentionally transparent and tunable.

`wrist_roll.pos` is driven by palm left-right tilt relative to neutral. `wrist_flex.pos` is driven by a pitch-like hand feature relative to neutral, such as the relationship between wrist, index MCP, and middle-finger landmarks. Exact landmark formulas can be adjusted during implementation, but they must be deterministic and documented in code.

`gripper.pos` is driven by continuous absolute pinch openness. Thumb-tip to index-tip distance is normalized against a hand-size estimate so the same gesture works at different distances from the camera. A full pinch maps to closed gripper. Open fingers map to open gripper. Intermediate pinch distance maps continuously between those states. Capturing neutral does not offset this gripper mapping; before enabling sync, the operator should hold a pinch state that matches the intended gripper target.

The mapper outputs desired targets in the same normalized position space used by LeRobot for the SO101 follower. Gains, clamps, smoothing coefficient, FPS, stale timeout, hand selection, and gripper open/closed calibration values should be CLI flags or a small config object so they can be tuned without changing mapping code.

## Robot Backend

Dry-run mode is the default and never opens the follower serial port.

Robot mode requires `--enable-robot`, a follower port, a robot id, and a calibration directory or existing calibration path compatible with the local LeRobot setup. On startup, the backend connects to `SO101Follower`, reads the current observation, and records startup positions for all six joints.

Every command action sent to the robot includes all action keys expected by `SO101Follower`, but only the wrist and gripper values change from the startup baseline. The held keys are:

- `shoulder_pan.pos`
- `shoulder_lift.pos`
- `elbow_flex.pos`

The controlled keys are:

- `wrist_flex.pos`
- `wrist_roll.pos`
- `gripper.pos`

The backend should use LeRobot's existing `max_relative_target` safety mechanism where practical, plus the prototype's own target filter before calling `send_action`.

## Safety Model

This is lower risk than full-arm teleoperation, but it is not inherently safe. Wrist and gripper movement can still pinch, collide with the table, tug cabling, or jump because of tracking noise. The design treats "no motion" as the default response to uncertainty.

Required safety behavior:

- Dry-run by default.
- Physical motion only with `--enable-robot`.
- Startup paused.
- No command output before neutral capture.
- Sync toggle with `space`.
- Optional deadman key mode.
- Controlled joints limited to wrist flex, wrist roll, and gripper.
- Non-controlled joints held at startup positions.
- Conservative hard clamps around startup wrist positions and configured gripper bounds, with LeRobot's relative target guard also enabled in robot mode.
- Per-frame max delta limit per controlled joint.
- Exponential smoothing on targets or landmark-derived features.
- Freeze output when hand tracking is lost, stale, or below confidence.
- Clear status for paused, neutral missing, tracking lost, clamped, and robot enabled states.
- Clean disconnect on exit.

## CLI Shape

The exact names can be refined during implementation, but the CLI should support these concepts:

- `--check`: import dependencies, ensure the MediaPipe model is available, validate config, and exit without opening the camera or moving hardware.
- `--camera-index`, `--width`, `--height`, `--max-hands`: camera and MediaPipe controls inherited from the existing demo.
- `--enable-robot`: opt into physical robot commands.
- `--robot-port`, `--robot-id`, `--calibration-dir`: SO101 follower connection.
- `--fps`: control-loop rate cap.
- `--deadman-key`: require recent key activity for command output.
- `--wrist-flex-gain`, `--wrist-roll-gain`: mapping gains.
- `--gripper-open`, `--gripper-closed`: gripper target calibration.
- `--max-delta`: per-frame target movement limit.
- `--smoothing`: exponential smoothing coefficient.
- `--stale-timeout-ms`: hand-result freshness limit.

## Error Handling

Camera availability is checked before connecting to the robot. If the camera cannot open, the script exits without touching the robot.

Robot configuration is validated before entering the loop in robot mode. Missing port, missing id, or missing calibration should fail with an actionable message.

If MediaPipe has not produced a fresh confident hand result, the current safe target is frozen. If neutral has not been captured, command output remains disabled. If a computed target exceeds configured limits, it is clamped and the status should make that visible.

## Verification

Implementation should include focused tests for the pure logic:

- `pose_mapper.py`: synthetic landmark sets for open hand, full pinch, neutral roll, positive roll, negative roll, positive flex, and negative flex.
- `safety.py`: hard clamps, smoothing behavior, stale-result freeze, and max per-frame delta.

Manual verification should proceed in this order:

1. `--check` mode.
2. Dry-run webcam mode with neutral capture and visible target changes.
3. Robot mode with a low FPS, conservative clamps, neutral capture, and sync initially paused.
4. Robot mode with optional deadman key if the first physical run needs tighter control.

The first physical run should start with the wrist clear of the table and cables, gripper away from hands, and one hand ready to pause or quit.
