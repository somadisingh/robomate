# Copilot Hackathon Backend Analysis Pipeline Design

Date: 2026-05-24

## Goal

Consolidate the video-analysis playground experiments into a demo-ready backend pipeline in `backend/`, using Modal serverless compute plus Supabase Edge Functions, Storage, and Postgres.

When a collector uploads an iOS recording bundle, the backend should:

- preserve the existing upload flow from the iOS app into the private Supabase `recordings` bucket
- score the recording with Gemini 3.5 Flash and expose the score to the collector as soon as scoring finishes
- run hand landmarks, object detection, SAM instance segmentation, and temporal action segmentation in the background
- store compact review fields in Postgres and heavy visual artifacts in Storage
- provide enough status and artifact metadata for a minimal web results view now and a richer data-studio UI later
- include an end-to-end local CLI that uploads a real iPhone bundle fixture and waits for scoring or all analysis jobs

This is a hackathon backend. It should be quick to implement and reliable enough for the demo, without building a full production queue system.

## Non-Goals

This design does not implement the full data-studio UI, detailed overlay UX, point-cloud rendering, 3D Gaussian splatting, payout logic, or production-grade multi-tenant operations. The minimal web work is limited to showing backend status and compact results.

The existing deployed Supabase Edge Functions for TwelveLabs or natural-language video search are out of scope. The functions currently named `index-video`, `describe-video`, `task-status`, and `search-videos` should be left untouched because they belong to a separate search experiment.

## Current Context

The iOS app already writes recording bundles with this shape:

- `video.mp4`
- `imu.jsonl`
- `poses.jsonl`
- `intrinsics.json`
- optional `depth.bin`
- `metadata.json`

The `playground/data/iphone-data-2` folder is a representative 53 MB iOS bundle and should become the default E2E fixture.

The local `supabase/functions/submit-recording` function currently verifies the collector JWT, writes `recordings` and `submissions`, and returns quickly. The live Supabase schema has `recordings.is_scoring`, `recordings.success`, `recordings.success_reasoning`, `recordings.score`, and `recordings.score_reasoning`. The intended `recordings.summary` column should be used if present and added during implementation if it is still missing.

The `playground/modal-inference` package already validates separate Modal classes for YOLO26, SAM 3.1, MediaPipe Hands, and temporal action segmentation. The `playground/video_eval` package already validates Gemini Files API based scoring.

## Recommended Architecture

Use one Modal orchestrator entrypoint per recording, with durable per-analyzer status rows in Supabase.

This balances hackathon speed and reliability:

- one deployed Modal endpoint is easier than a full queue worker
- Supabase records visible status and errors for each analyzer
- the pipeline can be rerun idempotently by `recording_id`
- Gemini scoring can complete and publish before the heavier visual analyzers finish

The main components are:

- `supabase/functions/submit-recording`: upload finalizer and pipeline kickoff
- `backend/`: production Modal package promoted from the playground experiments
- Modal orchestrator: trusted backend worker that reads from and writes to Supabase
- analyzer modules: Gemini scoring, MediaPipe hands, YOLO objects, SAM segments, temporal actions
- minimal web results view: status and compact result consumer only
- E2E upload CLI: local test runner for real iOS bundles

Modal should run with narrowly scoped secrets for Supabase and model/API providers. It may use the Supabase service role because it needs to write analyzer results and Storage artifacts independent of the collector session. Service-role credentials must remain server-side only.

## Data Model

Reuse existing `recordings` scoring columns:

- `summary text`
- `success boolean`
- `success_reasoning text`
- `score numeric`
- `score_reasoning text`
- `is_scoring boolean`

`is_scoring` is the iOS-facing readiness flag. It means only "Gemini score is still pending." It does not mean the full analysis pipeline is still running.

Add or use these backend-analysis fields:

- `recordings.status text`: coarse pipeline state. Use `uploaded`, `analyzing`, `analyzed`, and `analysis_failed`.
- `recordings.detected_objects jsonb`: compact YOLO object summary for quick review and filtering.
- `recordings.analysis_artifacts jsonb`: stable paths to heavy analysis artifacts in Storage.
- `tasks.objects text[] not null default '{}'`: lab-provided object tags for SAM. Modal always unions this list with `human hand`.

Add `recording_analysis_jobs`:

- `id uuid primary key default gen_random_uuid()`
- `recording_id uuid not null references recordings(id)`
- `kind text not null`
- `status text not null`
- `artifact_path text`
- `summary jsonb`
- `error text`
- `started_at timestamptz`
- `finished_at timestamptz`
- `created_at timestamptz not null default now()`
- unique key on `(recording_id, kind)`

Supported job kinds:

- `gemini_eval`
- `mediapipe_hands`
- `yolo_objects`
- `sam_segments`
- `temporal_actions`

Supported job statuses:

- `pending`
- `running`
- `succeeded`
- `failed`

For the demo, these status strings can be plain text with check constraints rather than custom enum types.

## Storage Contract

Keep heavy visual data in the existing private `recordings` bucket under each recording folder:

- `<recording_id>/analysis/gemini-eval.json`
- `<recording_id>/analysis/mediapipe-hands.json`
- `<recording_id>/analysis/yolo-detections.json`
- `<recording_id>/analysis/sam-segments.json`
- `<recording_id>/analysis/temporal-actions.json`

The JSON artifacts should preserve enough frame-index, timestamp, image-size, bbox, landmark, polygon, mask, confidence, and segment metadata for a later web data studio to render overlays and timelines.

The DB should not store every bbox, hand joint, SAM polygon, or temporal segment. It should store only summaries and artifact paths.

## Upload And Analysis Flow

1. The iOS app streams bundle files to Supabase Storage under `<recording_id>/`.
2. The iOS app calls `submit-recording` with task, storage, stream, device, duration, size, and GPS metadata.
3. `submit-recording` verifies the collector JWT.
4. `submit-recording` upserts `recordings`, inserts `submissions`, sets `recordings.status = 'analyzing'`, and sets `is_scoring = true`.
5. `submit-recording` creates five `recording_analysis_jobs` rows, one per analyzer.
6. `submit-recording` invokes one Modal orchestrator endpoint with `recording_id`, `task_id`, `submission_id`, and `storage_path`.
7. `submit-recording` returns quickly to iOS. If Modal kickoff fails after the upload is saved, the response should still make clear that the upload succeeded and analysis did not start.
8. Modal fetches the task, recording metadata, and recording bundle.
9. Modal runs the analyzers concurrently where practical.
10. Each analyzer marks its job `running`, writes its artifact, then marks its job `succeeded` or `failed`.
11. Gemini writes `summary`, `success`, `success_reasoning`, `score`, and `score_reasoning` as soon as it finishes, then sets `is_scoring = false`.
12. MediaPipe, YOLO, SAM, and temporal jobs may continue after scoring is visible to the collector.
13. Modal updates `recordings.status` to `analyzed` when every job succeeds, or `analysis_failed` when all jobs finish and at least one failed.

## Analyzer Contracts

Gemini scoring:

- input: `video.mp4` and task description
- model: Gemini 3.5 Flash through the Files API path
- output fields: `summary`, `success`, `success_reasoning`, `score`, `score_reasoning`
- artifact: full JSON response at `analysis/gemini-eval.json`
- readiness: controls `is_scoring`

MediaPipe hands:

- input: `video.mp4`
- output artifact: sampled frame records with source frame, time, handedness, normalized landmarks, world landmarks when available, and confidence
- summary: frame count, sampled FPS, visible-hand ratio

YOLO objects:

- input: `video.mp4`
- output artifact: per-frame detections with bbox, class id, class name, and confidence
- DB summary: distinct detected object classes, counts, max confidence, and representative frames in `recordings.detected_objects`

SAM segments:

- input: `video.mp4`
- prompt tags: `tasks.objects` plus `human hand`
- output artifact: per-frame instance boxes and mask geometry suitable for overlay rendering
- summary: prompts used, per-prompt instance counts, frame count

Temporal action segmentation:

- input: `video.mp4`
- output artifact: action segments with start/end seconds, frame spans, hand, motion metrics, label/instruction metadata if available, and confidence when available
- summary: segment count and top labels

## E2E Test Tool

Add a local CLI in `backend/` that exercises the real upload path with a local bundle:

```bash
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle \
  --bundle playground/data/iphone-data-2 \
  --task-id 106760b6-43ec-41bd-b6f6-340b00db1d58 \
  --wait score
```

The CLI should:

- generate a fresh `recording_id` by default
- upload all bundle files to `recordings/<recording_id>/...`
- patch `metadata.json` in memory so `recordingId` and `bountyId` match the test run
- call `submit-recording` with the same payload shape the iOS app uses
- print `recording_id`, Storage paths, submission details when available, and analysis job statuses
- support `--wait score` to poll until `is_scoring = false` and then print the score fields
- support `--wait all` to poll until all `recording_analysis_jobs` are terminal and verify artifact paths exist

Authentication should be pragmatic:

- primary mode: collector email/password or an existing collector access token
- fallback smoke mode: service-role upload/insert plus direct Modal orchestrator invocation, used only when auth is blocking backend debugging

The fallback is acceptable for hackathon debugging, but the primary E2E path should mirror iOS as closely as possible.

## Minimal Web View

The first web slice should be operational, not a polished data studio.

For a lab reviewing a recording or task submission, show:

- original video preview
- `recordings.status`
- per-analyzer job statuses and errors
- Gemini summary, success, score, and reasoning
- detected object summary
- links or simple JSON previews for heavy artifacts

Do not build the final overlay/timeline/3D sensor UI in this spec. The Storage artifact contract is the foundation for that later work.

## Error Handling

Analyzer failures should be isolated:

- each analyzer updates only its own job status and error
- Gemini failure sets `is_scoring = false`, leaves score fields null, and records the scoring error in its job row
- non-Gemini failures do not block Gemini score visibility
- non-Gemini failures can make the final `recordings.status` become `analysis_failed`

Modal orchestrator reruns should be idempotent by `recording_id`:

- existing job rows should be reused
- artifacts can be overwritten in the same Storage paths
- compact `recordings` fields should be replaced with the latest successful result

If Modal kickoff fails in `submit-recording`, the upload should remain saved. The function should return an understandable response that says analysis did not start, so the E2E CLI and web view can show the failure.

## Verification

Implementation should include these verification paths:

- schema check: required columns and `recording_analysis_jobs` exist
- Modal dry run: analyzer modules import and can run against a local sample without writing production rows
- E2E upload score path: `playground/data/iphone-data-2` uploads, `submit-recording` runs, and `is_scoring` flips false after Gemini writes score fields
- E2E all path: all analyzer jobs reach terminal states and expected artifact paths exist
- failure injection: one non-Gemini analyzer fails while Gemini still publishes score fields

The E2E CLI is the primary demo confidence tool.

## Security And Demo Debt

The `recordings` bucket is private, but current Storage and table policies should be reviewed before production hardening. The live project also has `public.video_embeddings` with Row Level Security disabled. This design does not change that table because it belongs to the separate video-search work, but the issue should be resolved before exposing search broadly.

For this hackathon pipeline, the important constraints are:

- service-role keys only in Edge Functions, Modal secrets, or local `.env` files excluded from git
- no service-role key in iOS or browser code
- minimal RLS changes needed for the demo path
- analyzer artifacts remain in the private `recordings` bucket and are served through signed URLs or server-side reads

