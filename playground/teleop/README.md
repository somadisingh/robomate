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
