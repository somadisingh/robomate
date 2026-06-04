# Copilot Hackathon Backend

Modal serverless backend for recording analysis.

The deployed Modal app is `copilot-hackathon-backend-analysis`.

## Setup

```bash
cd backend
uv run --python 3.12 pytest
uv run --python 3.12 modal setup
```

Required local or Modal secret values:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_ANON_KEY`
- `GEMINI_API_KEY`
- `MODAL_ANALYSIS_SECRET`
- `GEMINI_MODEL` (optional; defaults to `gemini-3.5-flash`)
- `COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS` (optional; defaults off. Set to `1` for demo runs to enable YOLO, MediaPipe, SAM, temporal actions, and Gaussian splats.)

Deploy:

```bash
cd backend
uv run --python 3.12 modal deploy modal_app.py
```

The Supabase `submit-recording` function needs matching Edge Function secrets:

- `MODAL_ANALYSIS_URL`: `https://the-duck-company--copilot-hackathon-backend-analysis-submit-analysis.modal.run`
- `MODAL_ANALYSIS_SECRET`: same value as the Modal secret

## E2E Upload

Run the fixture through Supabase Storage, `submit-recording`, and the deployed
Modal pipeline:

```bash
cd backend
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle \
  --bundle ../playground/data/iphone-data-2 \
  --task-id 106760b6-43ec-41bd-b6f6-340b00db1d58 \
  --wait score
```

Use `--wait all` before a demo to verify every analyzer has completed and that
each artifact can be downloaded:

```bash
cd backend
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle \
  --bundle ../playground/data/iphone-data-2 \
  --task-id 106760b6-43ec-41bd-b6f6-340b00db1d58 \
  --wait all \
  --timeout-s 1800
```

`--wait score` returns as soon as Gemini scoring has populated
`recordings.summary`, `recordings.success`, `recordings.success_reasoning`,
`recordings.score`, `recordings.score_reasoning`, and flips
`recordings.is_scoring` to `false`.

`--wait all` waits for these job rows in `recording_analysis_jobs`:

- `gemini_eval`
- `mediapipe_hands`
- `yolo_objects`
- `sam_segments`
- `temporal_actions`

Heavy outputs are stored in the private `recordings` bucket under:

```text
{recording_id}/analysis/gemini-eval.json
{recording_id}/analysis/mediapipe-hands.json
{recording_id}/analysis/yolo-detections.json
{recording_id}/analysis/sam-segments.json
{recording_id}/analysis/temporal-actions.json
```
