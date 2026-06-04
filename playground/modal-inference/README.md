# Modal Inference Playground

Deployable Modal wrappers for the video-analysis experiments in `playground/`:

- Ultralytics YOLO26 object detection
- Ultralytics YOLO26 instance segmentation
- SAM 3.1 concept instance segmentation
- MediaPipe hand landmark estimation
- Temporal action segmentation from the local playground package

The app is intentionally split into separate Modal classes so CPU-only work does
not reserve a GPU, and the heavier SAM/YOLO models only load in containers that
need them.

## Setup

```bash
cd playground/modal-inference
uv run --python 3.12 modal setup
```

Deploy:

```bash
uv run --python 3.12 modal deploy modal_app.py
```

Run one remote call without deploying permanently:

```bash
uv run --python 3.12 modal run modal_app.py \
  --kind yolo \
  --media-path ../data/1_can_noodles.mp4 \
  --task detect \
  --max-frames 120 \
  --output-json outputs/yolo-detect.json
```

SAM 3.1 needs Meta/Hugging Face checkpoint access. Upload the approved
checkpoint to the shared model Volume before using `--kind sam`:

```bash
uv run --python 3.12 modal volume put copilot-hackathon-modal-inference-models \
  /path/to/sam3.pt \
  /sam/sam3.pt
```

Then run:

```bash
uv run --python 3.12 modal run modal_app.py \
  --kind sam \
  --media-path ../data/1_can_noodles.mp4 \
  --prompts "can,cup,hand" \
  --max-frames 120 \
  --output-json outputs/sam.json
```

MediaPipe hands:

```bash
uv run --python 3.12 modal run modal_app.py \
  --kind hands \
  --media-path ../data/1_can_noodles.mp4 \
  --target-fps 10 \
  --output-json outputs/hands.json
```

Temporal action segmentation:

```bash
uv run --python 3.12 modal run modal_app.py \
  --kind temporal \
  --media-path ../data/1_can_noodles.mp4 \
  --target-fps 10 \
  --output-json outputs/temporal.json
```

## Cost Controls

- YOLO and SAM request one GPU only, prefer smaller GPUs, and cap autoscaling.
- MediaPipe and temporal segmentation run CPU-only.
- `scaledown_window` is short and `min_containers` is left at the Modal default,
  so idle containers do not stay warm for long.
- Models are cached in the `copilot-hackathon-modal-inference-models` Volume instead of
  redownloading on every cold start.
- CLI examples use `--max-frames`; keep that during comparison runs, then remove
  or raise it for full videos.
- Responses return JSON boxes/landmarks/polygons, not rendered videos, to avoid
  large egress and serialization costs.

## Credentials

No API key is required for YOLO26, MediaPipe, or the offline temporal
segmentation path.

SAM 3.1 requires access to the Meta checkpoint. The current code expects the
checkpoint file in the Modal Volume at `/models/sam/sam3.pt` by default. If you
want the app to download that gated checkpoint automatically, provide a Modal
secret with an approved Hugging Face token and wire it into the SAM image.

The temporal action segmentation wrapper uses `labeler="none"` for inference.
If you want VLM captions in Modal, we should add an OpenAI secret to that class.
