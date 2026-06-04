# 3D Gaussian Splatting Pipeline — Design

**Status:** Approved (brainstorming) — pending implementation plan
**Author:** Chris Yoo
**Date:** 2026-05-24
**Spec path:** `docs/superpowers/specs/2026-05-24-gaussian-splat-pipeline-design.md`

## Goal

Deploy the iPhone RGB + LiDAR + IMU → 3D Gaussian Splatting pipeline (currently a research prototype in `playground/iphone_sensors/`) as a first-class analyzer in the recording analysis pipeline, with artifacts stored in Supabase Storage and an interactive 3D viewer in the web studio that supports camera-POV playback synced to the 2D video timeline.

## Scope decisions (locked in brainstorming)

| Decision | Choice |
| --- | --- |
| Trigger | **Auto on every submission**, fan-out with other analyzers |
| Artifact model | **Manifest JSON** (`artifact_path` points at it; manifest references siblings) |
| Web view model | **New "3D scene" view mode**, switchable from the 2D layered viewer |
| Playback sync | **Time-synced by default + free-camera toggle** |
| Backend layout | **Inline in `backend/modal_app.py`** alongside existing analyzers, own image |
| Training cost | **Match playground quality**: 7000 iters splatfacto on **A10G** GPU |
| Missing prerequisites | **Skip silently** (preflight check; no job row written) |

## Non-goals (v1)

- Re-train / quality-tier UI in the studio. (Re-running is supported by deleting the job row and resubmitting.)
- COLMAP / SfM-only fallback when LiDAR is missing.
- Per-frame edit / pose-graph optimization controls.
- Sharing or exporting splats outside the studio.

## Architecture overview

```
Recording lands → orchestrator process_recording (backend/modal_app.py)
   ├── preflight: depth_frame_count ≥ MIN_FRAMES && depth_width set
   │     └── if false: skip enqueueing gaussian_splat (no job row, no error)
   ├── mark_job(gaussian_splat, status='running')
   ├── SplatfactoTrainer().train.remote(recording_id, storage_path)
   │     ├── pulls video.mp4, depth.bin, poses.jsonl, intrinsics.json from Supabase
   │     ├── builds nerfstudio dataset in-container (port of export_nerfstudio.py)
   │     ├── ns-train splatfacto (7k iters) → ns-export gaussian-splat
   │     ├── PLY → .spz via embedded sparkjs Node CLI
   │     ├── builds camera_path.json (frame ↔ camera transform map)
   │     ├── uploads splat.spz, camera_path.json, seed_points.ply, manifest.json,
   │     │       train_config.json to
   │     │   supabase://recordings/{recording_id}/analysis/gaussian_splat/
   │     └── returns { manifest_path, summary }
   └── finalize_multi_artifact_job(api, request, 'gaussian_splat', manifest_path, summary)

Web (studio/[id]):
   page.tsx (server)
     └── if gaussian_splat job succeeded: fetch manifest, mint signed URLs for siblings
   studio.tsx (client)
     └── view-mode toggle: [ 2D layered ] [ 3D scene ]
            └── 3D mode mounts <SceneView /> (dynamic import of three + sparkjs)
                   ├── layer toggles: Splats | Camera path | Seed points
                   └── camera mode: Synced (driven by video currentTime) | Free (OrbitControls)
```

## Backend changes

### Files affected

- `backend/backend/contracts.py`
  - Add `'gaussian_splat'` to `AnalysisKind` literal and `ANALYSIS_KINDS` tuple.
- `backend/backend/artifacts.py`
  - Extend `analysis_artifact_paths()` to map `'gaussian_splat'` → `{recording_id}/analysis/gaussian_splat/manifest.json`.
  - Add helper `gaussian_splat_dir(recording_id) -> str` for directory prefix.
- `backend/backend/orchestrator.py`
  - Add `finalize_multi_artifact_job(api, request, kind, manifest_path, summary)` — sibling to `run_remote_analyzer` for multi-file outputs. Splat Modal function uploads its own files; orchestrator only updates the job row.
  - Add `gaussian_splat_preflight(recording) -> bool` — true iff depth metadata is set and `depth_frame_count >= MIN_FRAMES` (default 30).
- `backend/backend/splat/` (new package)
  - `dataset.py` — port of `playground/iphone_sensors/export_nerfstudio.py` (frame selection, transforms.json, optional depth maps), reading from in-memory bytes rather than disk paths to fit the Modal flow.
  - `convert.py` — PLY → SPZ via embedded Node + `@sparkjsdev/spark` (mirroring `modal_nerfstudio.py`'s conversion block).
  - `camera_path.py` — build `camera_path.json` from nerfstudio `transforms.json` + `dataparser_transforms.json`.
  - `manifest.py` — typed builder for `manifest.json`.
- `backend/backend/supabase_api.py`
  - Add `upload_bytes(bucket, path, data, content_type)` mirroring existing `upload_json` retry policy.
  - Add `download_bytes(bucket, path) -> bytes` (used by splat trainer for video/depth/poses).
- `backend/modal_app.py`
  - New `nerfstudio_image` — CUDA base, `nerfstudio>=1.1`, `gsplat`, `ffmpeg`, Node 20 + `@sparkjsdev/spark@2.1.0`, plus `backend` package (and `backend/backend/splat`).
  - New `@app.cls(gpu="A10G", image=nerfstudio_image, timeout=40*60)` class `SplatfactoTrainer` with `train(recording_id: str, storage_path: str) -> dict`.
  - In `process_recording`:
    - Call `gaussian_splat_preflight(recording)`.
    - If true: add `gaussian_splat` to the `REMOTE_ANALYSIS_KINDS` loop's `mark_job('running')` + submit `SplatfactoTrainer().train.remote(...)` to the executor; on completion use `finalize_multi_artifact_job`.
    - If false: no row, no future.

### Image build notes

`nerfstudio_image` is intentionally heavy (~CUDA + nerfstudio). It is **separate** from the existing fast-analyzer images so YOLO/SAM/MediaPipe rebuilds are unaffected. Modal's image cache means repeated deploys are cheap after the first build.

### Concurrency & timeouts

- The orchestrator's existing `ThreadPoolExecutor(max_workers=5)` is saturated by today's 5 analyzers. Bump to `max_workers=6` to accommodate `gaussian_splat`. The pool only holds orchestrator-side threads blocking on Modal futures, so the bump is essentially free.
- Heavy work executes on a GPU worker; the orchestrator thread costs nothing local.
- `process_recording` already has `timeout=45*60`. A10G splatfacto @ 7k iters is ~10–15 min, comfortable within budget.
- `SplatfactoTrainer.train` itself has `timeout=40*60` to leave headroom.

## Artifact format

### Storage layout

```
recordings/{recording_id}/analysis/gaussian_splat/
  manifest.json     ← job.artifact_path points here
  splat.spz         ← compressed gaussians (consumed by sparkjs)
  camera_path.json  ← per-frame camera transforms aligned to source video frames
  seed_points.ply   ← initial colored point cloud (rendered as a togglable layer)
  train_config.json ← nerfstudio config snapshot (debug/repro)
```

### `manifest.json` schema (version 1)

```json
{
  "version": 1,
  "splat": {
    "path": "splat.spz",
    "size_bytes": 12345678,
    "num_gaussians": 250000
  },
  "camera_path": {
    "path": "camera_path.json",
    "frame_count": 180,
    "fps": 30.0
  },
  "seed_points": { "path": "seed_points.ply" },
  "train": {
    "iterations": 7000,
    "gpu": "A10G",
    "duration_seconds": 612
  },
  "intrinsics": {
    "fx": 1462.0, "fy": 1462.0,
    "cx": 960.0, "cy": 540.0,
    "width": 1920, "height": 1080
  }
}
```

Paths are **relative** to the gaussian_splat directory. The web resolves them by joining with the dirname of `job.artifact_path`.

### `camera_path.json` schema

Array of `{ frame_index: int, time_seconds: float, position: [x,y,z], rotation_quaternion: [x,y,z,w] }`. Coordinates are in the nerfstudio-trained scene frame (the same frame `splat.spz` lives in).

### `summary` jsonb (for the job row)

Compact subset, lets the studio render a header without fetching the manifest:

```json
{
  "num_gaussians": 250000,
  "frame_count": 180,
  "train_duration_seconds": 612,
  "iterations": 7000
}
```

## Database migration

`supabase/migrations/<timestamp>_add_gaussian_splat_kind.sql`:

```sql
alter table public.recording_analysis_jobs
  drop constraint if exists recording_analysis_jobs_kind_check;

alter table public.recording_analysis_jobs
  add constraint recording_analysis_jobs_kind_check check (
    kind in (
      'gemini_eval',
      'mediapipe_hands',
      'yolo_objects',
      'sam_segments',
      'temporal_actions',
      'gaussian_splat'
    )
  );
```

No new columns required. The `unique (recording_id, kind)` constraint means re-running per-recording is a simple delete-then-resubmit.

## Web integration

### Files affected

- `web/src/app/studio/[id]/page.tsx`
  - Locate the `gaussian_splat` job from `analysisJobs`. If `status === 'succeeded'`, server-side fetch the manifest JSON via signed URL, resolve sibling paths, mint signed URLs for `splat.spz`, `camera_path.json`, `seed_points.ply` (1-hour expiry, matching existing convention).
  - Pass a typed `splatScene` prop into `studio.tsx`.
- `web/src/app/studio/[id]/studio.tsx`
  - Add a top-level view-mode toggle (`2D layered` ↔ `3D scene`). The 3D option is only enabled when `splatScene` is present.
  - In `3D scene` mode: hide the existing layer overlays, show the `<SceneView />` and `<SceneControls />` instead. The `<video>` element stays mounted (hidden, no controls) so its `currentTime` continues to drive the time-sync hook; the visible playback bar moves into the 3D pane and is bound to the same video element.
- `web/src/app/studio/[id]/scene-view.tsx` *(new, client component)*
  - Dynamic import of `three` and `@sparkjsdev/spark` (no SSR).
  - Owns: canvas ref, three.js scene/renderer/camera lifecycle, SparkRenderer instance, OrbitControls, RAF loop.
  - Mount: fetch `.spz` and `camera_path.json`; load into Spark; draw the camera trajectory as a `THREE.Line` along the path positions; load the PLY as a `THREE.Points` cloud (hidden by default).
  - Exposes a small imperative API consumed by `use-camera-sync.ts` for setting camera transform per frame.
- `web/src/app/studio/[id]/scene-controls.tsx` *(new, client component)*
  - Toggle group: Splats / Camera trajectory / Seed points.
  - Toggle: Camera mode — Synced (default) / Free.
- `web/src/app/studio/[id]/use-camera-sync.ts` *(new)*
  - Subscribes to the shared video element's `timeupdate` events.
  - `frameIdx = clamp(floor(currentTime * fps), 0, frame_count - 1)`.
  - Writes camera position + quaternion from `camera_path.json[frameIdx]` to the three.js camera.
  - Disabled when free-camera mode is selected (releases control to OrbitControls).
- `web/package.json`
  - Add `three` and `@sparkjsdev/spark` at the versions used by the playground (`2.1.0` for spark; latest stable three matching spark's peer range).

### Time-sync semantics

- The 2D layered viewer is the canonical playback clock — its `<video>` element drives everything.
- In v1, no interpolation between frames; per-frame snap is acceptable since camera_path is one entry per source frame.
- `fps` is read from the manifest (computed in the trainer from `frame_count / duration` of the selected nerfstudio dataset).

### Bundle hygiene

Three.js + sparkjs ship only when 3D mode is activated. Studio entry bundle stays unchanged for non-3D users.

## Failure modes & observability

| Case | Behavior |
| --- | --- |
| Missing/short depth at submit | Preflight returns false; no job row written; studio never shows the 3D toggle. Single info log line server-side. |
| Splatfacto crashes / runs out of memory | Existing `as_completed` handler marks the job `failed` with the exception string; studio shows "3D scene generation failed" disabled toggle. |
| Storage upload fails | `upload_bytes` retry on 5xx (≥3 attempts, exponential). If all fail, job marked failed. |
| Manifest fetch fails on web | Page falls back to 2D-only mode; renders an inline notice. |
| `.spz` fetch fails in browser | `<SceneView />` shows an inline error; user can switch back to 2D. |

No new metrics tables. Existing job-status fields suffice. (Future: training duration is in `summary.train_duration_seconds` if we want to chart it.)

## Testing

- `backend/tests/test_splat_manifest.py` — pure-Python tests for `manifest.py` and `camera_path.py` from a fixture `transforms.json`.
- `backend/tests/test_orchestrator_preflight.py` — preflight true/false for recordings with/without depth metadata.
- `backend/tests/test_splat_dataset.py` — `dataset.py` selects frames at expected stride from a synthetic 60-frame capture.
- Smoke under `playground/iphone_sensors/tests/` — keeps the existing playground training callable, asserts manifest schema after a short run if a sample dataset is present.
- Web: no automated tests for the three.js renderer (WebGL is impractical to unit-test). Manual QA checklist covers view toggle, layer toggles, time-sync, free-camera, fail states.

## Relationship to `playground/iphone_sensors/modal_nerfstudio.py`

Kept as the research surface. Both the playground entry and `backend/modal_app.py`'s `SplatfactoTrainer` call into the shared `backend/backend/splat/` modules (`dataset`, `convert`, `camera_path`, `manifest`). The playground continues to read/write the local Modal volume; production reads/writes Supabase. This lets us keep iterating on training quality independently of the prod path.

## Rollout

1. Migration → splat package + helpers (no orchestrator wiring).
2. `SplatfactoTrainer` Modal class + image, callable from `playground/iphone_sensors` or a one-shot test.
3. Orchestrator preflight + fan-out + `finalize_multi_artifact_job`.
4. Web: page.tsx manifest resolution; `<SceneView />` mount; controls; time-sync; bundle splitting.
5. End-to-end smoke on one real recording.
