# Temporal Action Segmentation Playground

Hackathon-scale, VITRA-inspired action labelling for unlabelled egocentric
video:

1. Track left/right hands with MediaPipe.
2. Smooth 2D palm trajectories.
3. Propose temporal boundaries at VITRA-style hand-speed minima.
4. Render contact sheets with trajectory overlays.
5. Optionally ask a vision model to reject or caption each clip.
6. Write JSONL plus a lightweight review HTML page.

This intentionally does not reproduce VITRA's full 3D reconstruction stack. It
borrows the useful decomposition: use motion to make clips atomic, then use a
VLM or human review to name them.

## Quick Start

```bash
cd playground/temporal_action_segmentation
uv run --python 3.12 tas --check
```

Run offline segmentation and contact-sheet export:

```bash
uv run --python 3.12 tas process --video ../data/example.mp4 --labeler none
```

Run every `.mp4` in a folder:

```bash
uv run --python 3.12 tas process --input-dir videos/raw --glob '*.mp4'
```

Enable VLM labelling after your OpenAI environment credentials are configured:

```bash
uv run --python 3.12 tas process \
  --video ../data/example.mp4 \
  --labeler openai \
  --openai-model gpt-5.4-mini
```

The CLI loads `playground/temporal_action_segmentation/.env` automatically when
present. `.env.example` defaults to `gpt-5.4-mini`, and `--openai-model`
overrides it for one run.

## Outputs

By default, outputs go to `outputs/temporal_action_segmentation`:

- `segments.jsonl`: one proposed clip per line.
- `review.html`: static review page with contact sheets and labels.
- `tracks/*.csv`: per-sampled-frame hand centers and confidences.
- `contact_sheets/*/*.jpg`: sampled frames with hand trajectory overlays.
- `plots/*_speed.jpg`: speed curve and proposed cut boundaries.
- `cache/*.json`: cached VLM responses when `--labeler openai` is used.

Each JSONL record looks like:

```json
{
  "video_id": "example",
  "video_path": "/abs/path/example.mp4",
  "start_sec": 12.4,
  "end_sec": 14.1,
  "hand": "right",
  "caption": "Pick up the mug",
  "meaningful_manipulation": true,
  "confidence": 0.82
}
```

## Tuning Knobs

- `--target-fps`: downsample before tracking; 8-12 fps is usually enough.
- `--min-seg-s` / `--max-seg-s`: enforce clip duration.
- `--min-visible-ratio`: discard clips where the target hand is mostly absent.
- `--min-motion`: discard low-motion hand tracks.
- `--frames-per-clip`: contact-sheet samples sent to the VLM.
- `--merge-same-caption`: merge adjacent clips with the same non-`N/A` caption.

The biggest weakness is egocentric camera motion. This playground tracks hand
motion in image space, so a fast head turn can create false motion. For a
hackathon, contact sheets and the review page are the intended correction loop.
