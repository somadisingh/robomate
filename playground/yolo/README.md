# YOLO26 Video Playground

Runs Ultralytics YOLO26 video inference for:

- object detection
- OBB object detection
- object tracking
- semantic segmentation
- instance segmentation

The runner reads `.mp4` files from `../data` and writes rendered videos to
`../outputs/yolo`.

```bash
uv run --python 3.12 python run_yolo.py
```

By default the script tries `mps` first on Apple Silicon, then CUDA, then CPU.
Override it when needed:

```bash
uv run --python 3.12 python run_yolo.py --device cpu
uv run --python 3.12 python run_yolo.py --tasks detect track --overwrite
```
