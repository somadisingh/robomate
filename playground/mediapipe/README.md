Run MediaPipe Hands against the default webcam with MediaPipe Tasks live-stream
mode:

```bash
uv run python main.py
```

The first run downloads the official MediaPipe Hand Landmarker model to
`models/hand_landmarker.task`. Press `q` or `Esc` to close the camera window.

Useful options:

```bash
uv run python main.py --camera-index 1
uv run python main.py --width 640 --height 480
uv run python main.py --check
```

On macOS, allow camera access for the terminal app if the camera does not open.

Sample videos live in the shared playground data directory. To render the hand
landmark overlays for every shared `.mp4` sample:

```bash
uv run python main.py --sample-videos
```

Rendered videos are written to `../outputs/mediapipe`.
