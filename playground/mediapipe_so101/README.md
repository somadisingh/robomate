# MediaPipe SO101 Wrist Teleop

Real-time MediaPipe hand-pose control for the SO101 follower wrist and gripper.

The script defaults to dry-run mode. It only opens the robot serial port when `--enable-robot` is passed.

## Setup

```bash
uv sync
```

## Check

```bash
uv run python main.py --check
```

## Dry Run

```bash
uv run python main.py --camera-index 0 --width 640 --height 480 --fps 15
```

Controls:

- `n`: capture neutral hand pose.
- `space`: toggle real-time sync.
- `q` or `Esc`: exit.

Before neutral capture, no targets are emitted. Neutral capture requires fresh, high-confidence hand tracking. In dry-run mode, targets are displayed in the camera overlay and no robot port is opened.

## Robot Mode

Check robot configuration without opening the camera:

```bash
uv run python main.py \
  --check \
  --enable-robot \
  --robot-port /dev/cu.usbmodemYOUR_PORT \
  --robot-id so101_5AE60843881 \
  --calibration-dir ../so101/calibration/robots/so_follower
```

Example:

```bash
uv run python main.py \
  --enable-robot \
  --robot-port /dev/cu.usbmodemYOUR_PORT \
  --robot-id so101_5AE60843881 \
  --calibration-dir ../so101/calibration/robots/so_follower \
  --fps 10 \
  --max-delta 2.0 \
  --wrist-flex-limit 15 \
  --wrist-roll-limit 25
```

Use `--deadman-key x` to require deadman key activity for command output. Each detected keypress is active for `--deadman-grace-ms` milliseconds, default `175`; press repeatedly, or hold if your OS/window key repeat is active.

```bash
uv run python main.py \
  --enable-robot \
  --robot-port /dev/cu.usbmodemYOUR_PORT \
  --robot-id so101_5AE60843881 \
  --calibration-dir ../so101/calibration/robots/so_follower \
  --deadman-key x
```

## Mapping

- Palm left-right tilt maps to `wrist_roll.pos`.
- Hand flex relative to neutral maps to `wrist_flex.pos`.
- Thumb-index pinch distance maps continuously to `gripper.pos`.
- Full pinch means closed gripper.
- Open fingers mean open gripper.

Neutral capture affects wrist deltas only. The gripper command is absolute from current pinch openness, so when sync turns on the gripper will move toward the current pinch target rather than preserving the startup gripper position.

## Model Cache

The MediaPipe hand landmarker model is downloaded on first use and cached in `models/`. The cached `.task` and temporary download files are ignored by git.

## Safety Notes

This has a narrower command surface than full-arm teleoperation because it only commands wrist flex, wrist roll, and gripper. It is still physical robot motion. Keep the wrist clear of the table and cables, keep fingers out of the gripper, start with low FPS and small limits, and press `space` or `q` if motion is unexpected.

Before enabling sync, hold a pinch state that matches the desired gripper position. Tune `--gripper-open`, `--gripper-closed`, `--gripper-min`, `--gripper-max`, and `--max-delta` conservatively for the first physical run.

The script freezes command output when tracking is missing, stale, below confidence, paused, the deadman key is inactive, or neutral has not been captured. A backend send failure disables sync and locks command output off until restart.
