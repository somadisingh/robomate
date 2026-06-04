# Video Eval

Small Gemini-based evaluator for collected task trajectories. Given a video and
a task description, it returns structured JSON:

```json
{
  "task_succeeded": true,
  "success_reasoning": "The object is visibly moved to the requested target.",
  "trajectory_score": 8,
  "score_reasoning": "The trajectory succeeds with one small correction."
}
```

## Setup

```bash
cd playground/video_eval
uv run --python 3.12 pytest
```

Put your Gemini API key in `.env`:

```bash
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash
```

`GEMINI_MODEL` is optional. The CLI also accepts `--model` if the API exposes a
different Gemini 3.5 Flash alias for your account.

## Run

```bash
uv run --python 3.12 video-eval \
  --video ../data/1_can_noodles.mp4 \
  --task "Move the can next to the noodles."
```

Write the JSON result to a file:

```bash
uv run --python 3.12 video-eval \
  --video ../data/2_can_cup.mp4 \
  --task "Place the can into the cup." \
  --output outputs/2_can_cup_eval.json
```

Check the prompt/configuration without calling Gemini:

```bash
uv run --python 3.12 video-eval \
  --video ../data/3_plate_cutlery.mp4 \
  --task "Put the cutlery on the plate." \
  --dry-run
```

The tool uploads the video with the Gemini File API, waits for processing, asks
Gemini for JSON matching `VideoTaskEvaluation`, validates the response with
Pydantic, prints the result, and deletes the uploaded file by default.
