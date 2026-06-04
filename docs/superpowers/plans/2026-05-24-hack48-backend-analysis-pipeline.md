# Copilot Hackathon Backend Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a demo-ready Supabase plus Modal backend pipeline that scores uploaded iOS recording bundles quickly and fills in richer video-analysis artifacts in the background.

**Architecture:** The existing Supabase `submit-recording` Edge Function remains the upload finalizer, creates durable per-analyzer rows, and calls a Modal web endpoint. The Modal endpoint spawns a detached recording processor, which runs Gemini scoring and vision analyzers, stores heavy JSON artifacts in the private `recordings` bucket, and updates compact fields in Postgres. A local E2E CLI uploads `playground/data/iphone-data-2` through the same path as the iOS app.

**Tech Stack:** Supabase Postgres, Supabase Storage, Supabase Edge Functions on Deno, Modal Python serverless functions, Gemini Files API, MediaPipe, Ultralytics YOLO26/SAM, Next.js 16, TypeScript, Python 3.12, uv, pytest.

---

## Scope Check

This spec spans schema, Edge Functions, Modal, a local E2E test CLI, and a small web status view. These pieces are coupled by one recording-analysis workflow and should be implemented as one plan. Keep the implementation demo-oriented: no production queue runner, no polished data-studio UX, and no changes to deployed TwelveLabs/search Edge Functions.

## File Structure

Create or modify these files:

- Create: `backend/pyproject.toml` - uv project for the Modal backend package and local tooling.
- Create: `backend/README.md` - setup, secrets, deploy, and E2E commands.
- Create: `backend/modal_app.py` - Modal app, web endpoint, detached processor, and model classes promoted from the playground.
- Create: `backend/backend/__init__.py` - package marker.
- Create: `backend/backend/contracts.py` - Pydantic models and shared analyzer constants.
- Create: `backend/backend/artifacts.py` - artifact path builders and compact summary helpers.
- Create: `backend/backend/supabase_api.py` - small HTTP wrapper for Supabase REST and Storage using service-role or collector tokens.
- Create: `backend/backend/analyzers/__init__.py` - analyzer package marker.
- Create: `backend/backend/analyzers/gemini_eval.py` - Gemini Files API scoring contract with summary field.
- Create: `backend/backend/orchestrator.py` - orchestration logic independent of Modal decorators.
- Create: `backend/backend/tools/__init__.py` - tools package marker.
- Create: `backend/backend/tools/e2e_upload_bundle.py` - local E2E upload and polling CLI.
- Create: `backend/tests/test_artifacts.py` - artifact path and summary tests.
- Create: `backend/tests/test_gemini_eval.py` - Gemini prompt/schema tests without API calls.
- Create: `backend/tests/test_e2e_bundle.py` - metadata patch and upload payload tests.
- Create: `backend/tests/test_orchestrator.py` - orchestrator status-update tests with fakes.
- Create: `supabase/migrations/<generated>_backend_analysis_pipeline.sql` - generated with `supabase migration new backend_analysis_pipeline`.
- Modify: `supabase/functions/submit-recording/index.ts` - create jobs and kick off Modal.
- Modify: `web/src/app/lab/tasks/new/page.tsx` - persist task object tags for SAM.
- Modify: `web/src/app/lab/tasks/new/create-task-form.tsx` - simple object-tags input.
- Modify: `web/src/app/lab/tasks/[id]/page.tsx` - load recordings and analysis jobs for submissions.
- Modify: `web/src/app/lab/tasks/[id]/submissions-live.tsx` - display compact analysis status and scoring fields.

Do not modify `supabase/functions/index-video`, `describe-video`, `task-status`, or `search-videos`. They are deployed in the live project but not source-controlled here, and they belong to a separate natural-language video-search path.

### Modal API Notes

Use `@modal.fastapi_endpoint(method="POST")` for the web kickoff endpoint. Use `process_recording.spawn(payload)` inside that endpoint so the HTTP call returns quickly while the detached processor continues. Modal documents `fastapi_endpoint` for web endpoints and `Function.spawn()` for non-blocking function calls.

## Task 1: Backend Package, Contracts, And Pure Helpers

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/README.md`
- Create: `backend/backend/__init__.py`
- Create: `backend/backend/contracts.py`
- Create: `backend/backend/artifacts.py`
- Create: `backend/tests/test_artifacts.py`

- [ ] **Step 1: Create the backend uv project**

Create `backend/pyproject.toml` with:

```toml
[project]
name = "copilot-hackathon-backend"
version = "0.1.0"
description = "Modal backend pipeline for Copilot Hackathon recording analysis."
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "google-genai>=1.0.0",
    "httpx>=0.28.0",
    "modal>=1.1.0",
    "numpy>=2.2.0",
    "opencv-python-headless>=4.13.0.92",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.1",
]

[dependency-groups]
dev = [
    "pytest>=8.4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.uv]
python-preference = "only-system"
```

Create `backend/README.md` with:

```markdown
# Copilot Hackathon Backend

Modal serverless backend for recording analysis.

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

Deploy:

```bash
cd backend
uv run --python 3.12 modal deploy modal_app.py
```

Run the E2E upload fixture:

```bash
cd backend
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle \
  --bundle ../playground/data/iphone-data-2 \
  --task-id 106760b6-43ec-41bd-b6f6-340b00db1d58 \
  --wait score
```
```

- [ ] **Step 2: Add failing tests for artifact paths and summaries**

Create `backend/tests/test_artifacts.py` with:

```python
from backend.artifacts import (
    ANALYSIS_FILENAMES,
    analysis_artifact_paths,
    detected_object_summary,
    normalize_sam_prompts,
)


def test_analysis_artifact_paths_are_stable():
    paths = analysis_artifact_paths("abc123")

    assert paths == {
        "gemini_eval": "abc123/analysis/gemini-eval.json",
        "mediapipe_hands": "abc123/analysis/mediapipe-hands.json",
        "yolo_objects": "abc123/analysis/yolo-detections.json",
        "sam_segments": "abc123/analysis/sam-segments.json",
        "temporal_actions": "abc123/analysis/temporal-actions.json",
    }
    assert set(paths) == set(ANALYSIS_FILENAMES)


def test_detected_object_summary_compacts_frame_records():
    yolo_payload = {
        "frames": [
            {
                "frame_index": 0,
                "instances": [
                    {"class_name": "cup", "confidence": 0.8},
                    {"class_name": "cup", "confidence": 0.9},
                ],
            },
            {
                "frame_index": 8,
                "instances": [
                    {"class_name": "bottle", "confidence": 0.4},
                ],
            },
        ]
    }

    assert detected_object_summary(yolo_payload) == [
        {"class_name": "cup", "count": 2, "max_confidence": 0.9, "representative_frame": 0},
        {"class_name": "bottle", "count": 1, "max_confidence": 0.4, "representative_frame": 8},
    ]


def test_normalize_sam_prompts_always_includes_human_hand():
    prompts = normalize_sam_prompts(["Cup", " cup ", "", "can"])

    assert prompts == ["cup", "can", "human hand"]
```

- [ ] **Step 3: Run the tests and verify they fail**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.artifacts'`.

- [ ] **Step 4: Implement contracts and artifact helpers**

Create `backend/backend/__init__.py` with:

```python
"""Copilot Hackathon Modal backend package."""
```

Create `backend/backend/contracts.py` with:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AnalysisKind = Literal[
    "gemini_eval",
    "mediapipe_hands",
    "yolo_objects",
    "sam_segments",
    "temporal_actions",
]

AnalysisStatus = Literal["pending", "running", "succeeded", "failed"]

ANALYSIS_KINDS: tuple[AnalysisKind, ...] = (
    "gemini_eval",
    "mediapipe_hands",
    "yolo_objects",
    "sam_segments",
    "temporal_actions",
)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str
    task_id: str
    submission_id: str | None = None
    storage_path: str


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    description: str | None = None
    objects: list[str] = Field(default_factory=list)


class RecordingRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    storage_path: str
    streams: list[str] | None = None


class GeminiEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    success: bool
    success_reasoning: str = Field(min_length=1)
    score: int = Field(ge=0, le=10)
    score_reasoning: str = Field(min_length=1)
```

Create `backend/backend/artifacts.py` with:

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.contracts import AnalysisKind


ANALYSIS_FILENAMES: dict[AnalysisKind, str] = {
    "gemini_eval": "gemini-eval.json",
    "mediapipe_hands": "mediapipe-hands.json",
    "yolo_objects": "yolo-detections.json",
    "sam_segments": "sam-segments.json",
    "temporal_actions": "temporal-actions.json",
}


def analysis_artifact_paths(recording_id: str) -> dict[AnalysisKind, str]:
    prefix = recording_id.strip().strip("/")
    return {
        kind: f"{prefix}/analysis/{filename}"
        for kind, filename in ANALYSIS_FILENAMES.items()
    }


def normalize_sam_prompts(objects: list[str] | None) -> list[str]:
    seen: set[str] = set()
    prompts: list[str] = []
    for raw in [*(objects or []), "human hand"]:
        prompt = raw.strip().lower()
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def detected_object_summary(yolo_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "max_confidence": 0.0, "representative_frame": None}
    )
    for frame in yolo_payload.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        for instance in frame.get("instances", []):
            class_name = str(instance.get("class_name") or "unknown")
            confidence = float(instance.get("confidence") or 0.0)
            item = by_class[class_name]
            item["count"] += 1
            if confidence > item["max_confidence"]:
                item["max_confidence"] = round(confidence, 6)
                item["representative_frame"] = frame_index

    return [
        {"class_name": class_name, **values}
        for class_name, values in sorted(
            by_class.items(),
            key=lambda entry: (-entry[1]["count"], entry[0]),
        )
    ]
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_artifacts.py -q
```

Expected: PASS.

Commit:

```bash
git add backend/pyproject.toml backend/README.md backend/backend/__init__.py backend/backend/contracts.py backend/backend/artifacts.py backend/tests/test_artifacts.py
git commit -m "Add backend analysis contracts"
```

## Task 2: Supabase Schema Migration

**Files:**
- Create: `supabase/migrations/<generated>_backend_analysis_pipeline.sql`

- [ ] **Step 1: Create the migration file with Supabase CLI**

Run:

```bash
supabase migration new backend_analysis_pipeline
```

Expected: Supabase prints a new file under `supabase/migrations/` ending in `_backend_analysis_pipeline.sql`.

- [ ] **Step 2: Write the migration SQL**

Open the generated migration file and replace its contents with:

```sql
alter table public.recordings
  add column if not exists summary text,
  add column if not exists detected_objects jsonb,
  add column if not exists analysis_artifacts jsonb;

alter table public.recordings
  alter column is_scoring set default true;

alter table public.tasks
  add column if not exists objects text[] not null default '{}';

create table if not exists public.recording_analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  recording_id uuid not null references public.recordings(id) on delete cascade,
  kind text not null,
  status text not null default 'pending',
  artifact_path text,
  summary jsonb,
  error text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz not null default now(),
  unique (recording_id, kind),
  constraint recording_analysis_jobs_kind_check check (
    kind in (
      'gemini_eval',
      'mediapipe_hands',
      'yolo_objects',
      'sam_segments',
      'temporal_actions'
    )
  ),
  constraint recording_analysis_jobs_status_check check (
    status in ('pending', 'running', 'succeeded', 'failed')
  )
);

alter table public.recording_analysis_jobs enable row level security;

drop policy if exists "collectors read own recording analysis jobs"
  on public.recording_analysis_jobs;
create policy "collectors read own recording analysis jobs"
  on public.recording_analysis_jobs
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.recordings r
      where r.id = recording_analysis_jobs.recording_id
        and r.collector_id = auth.uid()
    )
  );

drop policy if exists "labs read task recording analysis jobs"
  on public.recording_analysis_jobs;
create policy "labs read task recording analysis jobs"
  on public.recording_analysis_jobs
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.recordings r
      join public.tasks t on t.id = r.bounty_id
      where r.id = recording_analysis_jobs.recording_id
        and t.lab_id = auth.uid()
    )
  );

create index if not exists recording_analysis_jobs_recording_id_idx
  on public.recording_analysis_jobs(recording_id);

create index if not exists recordings_bounty_id_idx
  on public.recordings(bounty_id);
```

- [ ] **Step 3: Apply the migration locally or to the linked dev project**

For local Supabase:

```bash
supabase db reset
```

For linked Supabase project during hackathon work:

```bash
supabase db push
```

Expected: migration applies without SQL errors.

- [ ] **Step 4: Verify schema through SQL**

Run through Supabase MCP or SQL editor:

```sql
select column_name
from information_schema.columns
where table_schema = 'public'
  and table_name = 'recordings'
  and column_name in ('summary', 'detected_objects', 'analysis_artifacts', 'is_scoring');

select column_name
from information_schema.columns
where table_schema = 'public'
  and table_name = 'tasks'
  and column_name = 'objects';

select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name = 'recording_analysis_jobs';
```

Expected: first query returns four recording columns, second returns `objects`, third returns `recording_analysis_jobs`.

- [ ] **Step 5: Commit**

Run:

```bash
git add supabase/migrations/*_backend_analysis_pipeline.sql
git commit -m "Add recording analysis schema"
```

## Task 3: Edge Function Kickoff

**Files:**
- Modify: `supabase/functions/submit-recording/index.ts`

- [ ] **Step 1: Update `submit-recording` to create durable jobs**

Modify `supabase/functions/submit-recording/index.ts` so the function:

- sets `recordings.status` to `analyzing`
- sets `recordings.is_scoring` to `true`
- inserts `submissions` with `.select("id").single()`
- upserts one row per analysis kind into `recording_analysis_jobs`
- calls Modal after the DB writes succeed

Use these constants near the top:

```ts
const analysisKinds = [
  "gemini_eval",
  "mediapipe_hands",
  "yolo_objects",
  "sam_segments",
  "temporal_actions",
] as const;
```

Use this Modal kickoff helper near the bottom:

```ts
async function startModalAnalysis(payload: {
  recording_id: string;
  task_id: string;
  submission_id: string | null;
  storage_path: string;
}) {
  const modalUrl = Deno.env.get("MODAL_ANALYSIS_URL");
  const modalSecret = Deno.env.get("MODAL_ANALYSIS_SECRET");
  if (!modalUrl || !modalSecret) {
    return { ok: false, error: "Modal analysis env is not configured" };
  }

  try {
    const res = await fetch(modalUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Copilot-Hackathon-Modal-Secret": modalSecret,
      },
      body: JSON.stringify(payload),
    });
    const text = await res.text();
    if (!res.ok) {
      return { ok: false, error: `Modal kickoff failed (${res.status}): ${text}` };
    }
    return { ok: true, body: text };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, error: `Modal kickoff failed: ${message}` };
  }
}
```

After the `recordings` upsert, create jobs with:

```ts
const jobRows = analysisKinds.map((kind) => ({
  recording_id: recordingId,
  kind,
  status: "pending",
  artifact_path: `${storagePath.replace(/\/$/, "")}/analysis/${
    kind === "gemini_eval" ? "gemini-eval.json"
      : kind === "mediapipe_hands" ? "mediapipe-hands.json"
      : kind === "yolo_objects" ? "yolo-detections.json"
      : kind === "sam_segments" ? "sam-segments.json"
      : "temporal-actions.json"
  }`,
  error: null,
  started_at: null,
  finished_at: null,
}));

const { error: jobsErr } = await admin
  .from("recording_analysis_jobs")
  .upsert(jobRows, { onConflict: "recording_id,kind" });
if (jobsErr) return json({ error: `recording_analysis_jobs: ${jobsErr.message}` }, 500);
```

Return a response shaped like:

```ts
return json({
  ok: true,
  recording_id: recordingId,
  submission_id: subData?.id ?? null,
  streams,
  analysis_started: modalResult.ok,
  analysis_error: modalResult.ok ? null : modalResult.error,
}, 200);
```

- [ ] **Step 2: Run a syntax check with Deno if available**

Run:

```bash
deno check supabase/functions/submit-recording/index.ts
```

Expected: PASS. If `deno` is not installed, run:

```bash
supabase functions serve submit-recording --no-verify-jwt
```

Expected: the function starts without TypeScript parse errors. Stop it after startup.

- [ ] **Step 3: Deploy only `submit-recording`**

Run:

```bash
supabase functions deploy submit-recording --project-ref coapgtbwmzxkfewzncxu
```

Expected: deployment succeeds and leaves the other deployed Edge Functions untouched.

- [ ] **Step 4: Commit**

Run:

```bash
git add supabase/functions/submit-recording/index.ts
git commit -m "Start Modal analysis from recording uploads"
```

## Task 4: Supabase API Wrapper And Gemini Scoring

**Files:**
- Create: `backend/backend/supabase_api.py`
- Create: `backend/backend/analyzers/__init__.py`
- Create: `backend/backend/analyzers/gemini_eval.py`
- Create: `backend/tests/test_gemini_eval.py`

- [ ] **Step 1: Write Gemini schema and prompt tests**

Create `backend/tests/test_gemini_eval.py` with:

```python
from backend.analyzers.gemini_eval import build_evaluation_prompt, response_schema
from backend.contracts import GeminiEvaluation


def test_prompt_includes_summary_and_task():
    prompt = build_evaluation_prompt("Pick up the cup.")

    assert "Pick up the cup." in prompt
    assert "summary" in prompt
    assert "score" in prompt
    assert "success_reasoning" in prompt


def test_gemini_evaluation_rejects_out_of_range_score():
    payload = {
        "summary": "The video shows a hand near a cup.",
        "success": True,
        "success_reasoning": "The cup is lifted.",
        "score": 11,
        "score_reasoning": "Too high.",
    }

    try:
        GeminiEvaluation.model_validate(payload)
    except Exception as exc:
        assert "score" in str(exc)
    else:
        raise AssertionError("Expected score validation to fail")


def test_response_schema_names_expected_fields():
    schema = response_schema()

    assert set(schema["properties"]) == {
        "summary",
        "success",
        "success_reasoning",
        "score",
        "score_reasoning",
    }
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_gemini_eval.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.analyzers'`.

- [ ] **Step 3: Implement Gemini analyzer**

Create `backend/backend/analyzers/__init__.py` with:

```python
"""Analysis modules used by the Modal orchestrator."""
```

Create `backend/backend/analyzers/gemini_eval.py` with:

```python
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from backend.contracts import GeminiEvaluation


DEFAULT_MODEL = "gemini-3.5-flash"


def build_evaluation_prompt(task_description: str) -> str:
    task = task_description.strip()
    if not task:
        raise ValueError("task_description must not be empty")

    return f"""Evaluate the attached video as a collected task trajectory.

Task description:
{task}

Return JSON only, matching this contract:
- summary: one brief sentence summarizing what is visible in the video.
- success: boolean. True only when the video visibly shows that the task goal was completed.
- success_reasoning: one or two brief sentences explaining the success decision.
- score: integer from 0 to 10 for the quality of the collected trajectory.
- score_reasoning: one or two brief sentences explaining the score.

Scoring guidance:
- 10: clean completion with efficient, stable, unambiguous trajectory.
- 7-9: task succeeds with minor inefficiency, hesitation, or correction.
- 4-6: partial progress or ambiguous success with notable trajectory issues.
- 1-3: little useful progress toward the task.
- 0: no relevant attempt or the task is impossible to evaluate from the video.

If the video is ambiguous, say so in the reasoning, set success to false unless the
success is visually clear, and choose a conservative score."""


def response_schema() -> dict[str, Any]:
    return GeminiEvaluation.model_json_schema()


def evaluate_video_file(
    *,
    video_path: str | Path,
    task_description: str,
    model: str | None = None,
    client: Any | None = None,
    cleanup_uploaded: bool = True,
    upload_timeout_s: float = 300,
    poll_interval_s: float = 2,
) -> GeminiEvaluation:
    resolved_video = Path(video_path).expanduser().resolve()
    if not resolved_video.is_file():
        raise FileNotFoundError(f"video file does not exist: {resolved_video}")

    active_model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    active_client = client or _make_gemini_client()
    uploaded_file = active_client.files.upload(file=str(resolved_video))

    try:
        uploaded_file = wait_for_uploaded_file(
            active_client,
            uploaded_file,
            timeout_s=upload_timeout_s,
            poll_interval_s=poll_interval_s,
        )
        response = active_client.models.generate_content(
            model=active_model,
            contents=[uploaded_file, build_evaluation_prompt(task_description)],
            config=_response_config(),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return GeminiEvaluation.model_validate_json(text)
    finally:
        if cleanup_uploaded:
            _delete_uploaded_file(active_client, uploaded_file)


def wait_for_uploaded_file(
    client: Any,
    uploaded_file: Any,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> Any:
    deadline = time.monotonic() + timeout_s
    current_file = uploaded_file
    while time.monotonic() < deadline:
        state = _state_name(current_file)
        if state in {"", "ACTIVE"}:
            return current_file
        if state == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {current_file.name}")
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
        current_file = client.files.get(name=current_file.name)
    raise TimeoutError(f"Timed out waiting for Gemini file processing: {uploaded_file.name}")


def _state_name(uploaded_file: Any) -> str:
    raw_state = getattr(uploaded_file, "state", None)
    if raw_state is None:
        return ""
    if hasattr(raw_state, "name"):
        return str(raw_state.name).upper()
    if hasattr(raw_state, "value"):
        return str(raw_state.value).split(".")[-1].upper()
    return str(raw_state).split(".")[-1].upper()


def _make_gemini_client() -> Any:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")
    return genai.Client(api_key=api_key)


def _response_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=response_schema(),
    )


def _delete_uploaded_file(client: Any, uploaded_file: Any) -> None:
    name = getattr(uploaded_file, "name", None)
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception:
        pass
```

- [ ] **Step 4: Implement the Supabase HTTP wrapper**

Create `backend/backend/supabase_api.py` with:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class SupabaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str

    @classmethod
    def from_service_role_env(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(url=url.rstrip("/"), key=key)

    @classmethod
    def from_anon_env(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required")
        return cls(url=url.rstrip("/"), key=key)


class SupabaseApi:
    def __init__(self, config: SupabaseConfig, *, bearer_token: str | None = None):
        self.config = config
        self.bearer_token = bearer_token or config.key
        self.client = httpx.Client(timeout=120)

    def close(self) -> None:
        self.client.close()

    def rest_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def storage_headers(self, content_type: str, *, upsert: bool = True) -> dict[str, str]:
        return {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": content_type,
            "x-upsert": "true" if upsert else "false",
        }

    def select_one(self, table: str, query: str) -> dict[str, Any]:
        url = f"{self.config.url}/rest/v1/{table}?{query}"
        res = self.client.get(url, headers=self.rest_headers({"Accept": "application/vnd.pgrst.object+json"}))
        return self._json(res)

    def patch_rows(self, table: str, query: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.config.url}/rest/v1/{table}?{query}"
        res = self.client.patch(
            url,
            headers=self.rest_headers({"Prefer": "return=representation"}),
            content=json.dumps(payload),
        )
        return self._json(res)

    def upsert_rows(self, table: str, payload: list[dict[str, Any]], *, on_conflict: str) -> list[dict[str, Any]]:
        url = f"{self.config.url}/rest/v1/{table}?on_conflict={on_conflict}"
        res = self.client.post(
            url,
            headers=self.rest_headers({"Prefer": "resolution=merge-duplicates,return=representation"}),
            content=json.dumps(payload),
        )
        return self._json(res)

    def upload_bytes(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        url = f"{self.config.url}/storage/v1/object/{bucket}/{path.lstrip('/')}"
        res = self.client.post(url, headers=self.storage_headers(content_type), content=data)
        self._raise_for_status(res)

    def upload_json(self, bucket: str, path: str, payload: Any) -> None:
        self.upload_bytes(
            bucket,
            path,
            json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            "application/json",
        )

    def download_bytes(self, bucket: str, path: str) -> bytes:
        url = f"{self.config.url}/storage/v1/object/{bucket}/{path.lstrip('/')}"
        res = self.client.get(url, headers=self.rest_headers())
        self._raise_for_status(res)
        return res.content

    def upload_file(self, bucket: str, path: str, file_path: Path, content_type: str) -> None:
        self.upload_bytes(bucket, path, file_path.read_bytes(), content_type)

    def _json(self, res: httpx.Response) -> Any:
        self._raise_for_status(res)
        if not res.content:
            return None
        return res.json()

    def _raise_for_status(self, res: httpx.Response) -> None:
        if res.status_code >= 400:
            raise SupabaseError(f"{res.request.method} {res.request.url} failed {res.status_code}: {res.text}")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_gemini_eval.py -q
```

Expected: PASS.

Commit:

```bash
git add backend/backend/supabase_api.py backend/backend/analyzers/__init__.py backend/backend/analyzers/gemini_eval.py backend/tests/test_gemini_eval.py
git commit -m "Add Gemini scoring backend"
```

## Task 5: Modal Orchestrator And Vision Analyzer Promotion

**Files:**
- Create: `backend/backend/orchestrator.py`
- Create: `backend/modal_app.py`
- Create: `backend/tests/test_orchestrator.py`
- Copy: `playground/modal-inference/modal_inference/` to `backend/backend/modal_inference/`
- Reference: `playground/modal-inference/modal_app.py`
- Reference: `playground/temporal_action_segmentation/temporal_action_segmentation/`

- [ ] **Step 1: Copy the validated Modal helper package**

Run:

```bash
mkdir -p backend/backend/modal_inference
cp playground/modal-inference/modal_inference/*.py backend/backend/modal_inference/
```

Expected: files such as `hand_landmarks.py`, `media.py`, and `ultralytics_results.py` exist under `backend/backend/modal_inference/`.

- [ ] **Step 2: Write orchestrator tests with fake dependencies**

Create `backend/tests/test_orchestrator.py` with:

```python
from backend.contracts import AnalysisRequest, GeminiEvaluation
from backend.orchestrator import apply_gemini_result, final_recording_status


class FakeSupabase:
    def __init__(self):
        self.patches = []

    def patch_rows(self, table, query, payload):
        self.patches.append((table, query, payload))
        return [payload]


def test_apply_gemini_result_flips_scoring_false():
    fake = FakeSupabase()
    result = GeminiEvaluation(
        summary="A hand picks up a cup.",
        success=True,
        success_reasoning="The cup is visibly lifted.",
        score=8,
        score_reasoning="The action succeeds with slight hesitation.",
    )

    apply_gemini_result(fake, "rec-1", result)

    assert fake.patches == [
        (
            "recordings",
            "id=eq.rec-1",
            {
                "summary": "A hand picks up a cup.",
                "success": True,
                "success_reasoning": "The cup is visibly lifted.",
                "score": 8,
                "score_reasoning": "The action succeeds with slight hesitation.",
                "is_scoring": False,
            },
        )
    ]


def test_final_recording_status_prefers_failed_when_any_job_failed():
    assert final_recording_status(["succeeded", "succeeded"]) == "analyzed"
    assert final_recording_status(["succeeded", "failed"]) == "analysis_failed"
    assert final_recording_status(["succeeded", "running"]) == "analyzing"


def test_analysis_request_contract():
    payload = AnalysisRequest(
        recording_id="r",
        task_id="t",
        submission_id="s",
        storage_path="r/",
    )

    assert payload.storage_path == "r/"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_orchestrator.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.orchestrator'`.

- [ ] **Step 4: Implement orchestration helpers**

Create `backend/backend/orchestrator.py` with:

```python
from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from backend.artifacts import analysis_artifact_paths, detected_object_summary, normalize_sam_prompts
from backend.analyzers.gemini_eval import evaluate_video_file
from backend.contracts import ANALYSIS_KINDS, AnalysisKind, AnalysisRequest, GeminiEvaluation, RecordingRecord, TaskRecord
from backend.supabase_api import SupabaseApi


AnalyzerFn = Callable[[bytes, str, TaskRecord, RecordingRecord], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def mark_job(api: SupabaseApi, recording_id: str, kind: AnalysisKind, status: str, **fields: Any) -> None:
    payload = {"status": status, **fields}
    api.patch_rows(
        "recording_analysis_jobs",
        f"recording_id=eq.{recording_id}&kind=eq.{kind}",
        payload,
    )


def apply_gemini_result(api: SupabaseApi, recording_id: str, result: GeminiEvaluation) -> None:
    api.patch_rows(
        "recordings",
        f"id=eq.{recording_id}",
        {
            "summary": result.summary,
            "success": result.success,
            "success_reasoning": result.success_reasoning,
            "score": result.score,
            "score_reasoning": result.score_reasoning,
            "is_scoring": False,
        },
    )


def final_recording_status(statuses: list[str]) -> str:
    if any(status in {"pending", "running"} for status in statuses):
        return "analyzing"
    if any(status == "failed" for status in statuses):
        return "analysis_failed"
    return "analyzed"


def fetch_context(api: SupabaseApi, request: AnalysisRequest) -> tuple[TaskRecord, RecordingRecord, bytes]:
    task = TaskRecord.model_validate(
        api.select_one("tasks", f"id=eq.{request.task_id}&select=*")
    )
    recording = RecordingRecord.model_validate(
        api.select_one("recordings", f"id=eq.{request.recording_id}&select=*")
    )
    video_path = request.storage_path.rstrip("/") + "/video.mp4"
    video_bytes = api.download_bytes("recordings", video_path)
    return task, recording, video_bytes


def run_gemini(api: SupabaseApi, request: AnalysisRequest, task: TaskRecord, video_bytes: bytes) -> dict[str, Any]:
    paths = analysis_artifact_paths(request.recording_id)
    mark_job(api, request.recording_id, "gemini_eval", "running", error=None)
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "video.mp4"
        video_path.write_bytes(video_bytes)
        result = evaluate_video_file(
            video_path=video_path,
            task_description=task.description or task.title,
        )
    payload = result.model_dump()
    api.upload_json("recordings", paths["gemini_eval"], payload)
    apply_gemini_result(api, request.recording_id, result)
    mark_job(
        api,
        request.recording_id,
        "gemini_eval",
        "succeeded",
        artifact_path=paths["gemini_eval"],
        summary=payload,
        finished_at=utc_now(),
    )
    return payload


def run_remote_analyzer(
    api: SupabaseApi,
    request: AnalysisRequest,
    kind: AnalysisKind,
    artifact_payload: dict[str, Any],
    db_summary: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> None:
    paths = analysis_artifact_paths(request.recording_id)
    api.upload_json("recordings", paths[kind], artifact_payload)
    mark_job(
        api,
        request.recording_id,
        kind,
        "succeeded",
        artifact_path=paths[kind],
        summary=db_summary,
        finished_at=utc_now(),
    )


def update_final_status(api: SupabaseApi, recording_id: str) -> None:
    rows = api.client.get(
        f"{api.config.url}/rest/v1/recording_analysis_jobs?recording_id=eq.{recording_id}&select=status",
        headers=api.rest_headers(),
    )
    statuses = [row["status"] for row in api._json(rows)]
    api.patch_rows("recordings", f"id=eq.{recording_id}", {"status": final_recording_status(statuses)})
```

- [ ] **Step 5: Implement Modal app by promoting existing classes**

Create `backend/modal_app.py` by starting from `playground/modal-inference/modal_app.py` and making these concrete changes:

1. Change app name to `copilot-hackathon-backend-analysis`.
2. Change imports from `modal_inference...` to `backend.modal_inference...`.
3. Keep `Yolo26`, `SAM31Segmenter`, `MediaPipeHands`, and `TemporalActionSegmenter` classes.
4. Add `timm` to `sam_image.pip_install(...)`.
5. Add a CPU image with `httpx`, `google-genai`, `pydantic`, and `python-dotenv`.
6. Add a `process_recording` Modal function that:
   - creates `SupabaseApi(SupabaseConfig.from_service_role_env())`
   - validates `AnalysisRequest`
   - fetches task, recording, and `video.mp4`
   - runs Gemini in one future
   - runs YOLO, SAM, MediaPipe, and temporal analyzers in other futures using the existing Modal classes
   - writes each artifact and job status as each future completes
   - calls `update_final_status(...)`
7. Add a `submit_analysis` web endpoint that validates `X-Copilot-Hackathon-Modal-Secret`, calls `process_recording.spawn(payload)`, and returns `{"ok": true, "call_id": call.object_id}`.

Use this endpoint skeleton:

```python
@app.function(
    image=orchestrator_image,
    secrets=[modal.Secret.from_name("copilot-hackathon-backend-secrets")],
    timeout=60,
    scaledown_window=30,
)
@modal.fastapi_endpoint(method="POST")
def submit_analysis(payload: dict, request):
    import os
    from fastapi import HTTPException

    expected = os.environ.get("MODAL_ANALYSIS_SECRET")
    received = request.headers.get("X-Copilot-Hackathon-Modal-Secret")
    if not expected or received != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    parsed = AnalysisRequest.model_validate(payload)
    call = process_recording.spawn(parsed.model_dump())
    return {"ok": True, "call_id": call.object_id}
```

Use this processor skeleton:

```python
@app.function(
    image=orchestrator_image,
    secrets=[modal.Secret.from_name("copilot-hackathon-backend-secrets")],
    timeout=45 * 60,
    scaledown_window=60,
    max_containers=4,
)
def process_recording(payload: dict) -> dict[str, object]:
    request = AnalysisRequest.model_validate(payload)
    api = SupabaseApi(SupabaseConfig.from_service_role_env())
    try:
        task, recording, video_bytes = fetch_context(api, request)
        api.patch_rows("recordings", f"id=eq.{request.recording_id}", {"status": "analyzing"})

        prompts = normalize_sam_prompts(task.objects)
        futures = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures[executor.submit(run_gemini, api, request, task, video_bytes)] = "gemini_eval"
            for kind in ("yolo_objects", "mediapipe_hands", "sam_segments", "temporal_actions"):
                mark_job(api, request.recording_id, kind, "running", error=None)
            futures[executor.submit(lambda: Yolo26().predict.remote(video_bytes, suffix=".mp4", task="detect", max_frames=None))] = "yolo_objects"
            futures[executor.submit(lambda: MediaPipeHands().landmarks.remote(video_bytes, suffix=".mp4", target_fps=10.0, max_frames=None))] = "mediapipe_hands"
            futures[executor.submit(lambda: SAM31Segmenter().segment.remote(video_bytes, suffix=".mp4", text_prompts=prompts, max_frames=None))] = "sam_segments"
            futures[executor.submit(lambda: TemporalActionSegmenter().segment.remote(video_bytes, suffix=".mp4", max_segments=200))] = "temporal_actions"

            for future in as_completed(futures):
                kind = futures[future]
                if kind == "gemini_eval":
                    future.result()
                    continue
                try:
                    payload = future.result()
                    summary = detected_object_summary(payload) if kind == "yolo_objects" else None
                    run_remote_analyzer(api, request, kind, payload, summary)
                    if kind == "yolo_objects":
                        api.patch_rows("recordings", f"id=eq.{request.recording_id}", {"detected_objects": summary})
                except Exception as exc:
                    mark_job(api, request.recording_id, kind, "failed", error=str(exc))

        update_final_status(api, request.recording_id)
        return {"ok": True, "recording_id": request.recording_id}
    finally:
        api.close()
```

- [ ] **Step 6: Run backend tests**

Run:

```bash
cd backend
uv run --python 3.12 pytest -q
```

Expected: PASS.

- [ ] **Step 7: Deploy Modal**

Run:

```bash
cd backend
uv run --python 3.12 modal deploy modal_app.py
```

Expected: deployment succeeds and prints or exposes a web endpoint URL for `submit_analysis`.

- [ ] **Step 8: Commit**

Run:

```bash
git add backend/modal_app.py backend/backend/modal_inference backend/backend/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "Add Modal recording analysis orchestrator"
```

## Task 6: E2E Upload Bundle CLI

**Files:**
- Create: `backend/backend/tools/__init__.py`
- Create: `backend/backend/tools/e2e_upload_bundle.py`
- Create: `backend/tests/test_e2e_bundle.py`

- [ ] **Step 1: Write metadata patch and payload tests**

Create `backend/tests/test_e2e_bundle.py` with:

```python
from backend.tools.e2e_upload_bundle import build_submit_payload, patch_metadata


def test_patch_metadata_sets_recording_and_task_ids():
    source = {
        "recordingId": "old",
        "bountyId": "old-task",
        "streams": ["video.mp4"],
    }

    patched = patch_metadata(source, recording_id="new-rec", task_id="new-task")

    assert patched["recordingId"] == "new-rec"
    assert patched["bountyId"] == "new-task"
    assert patched["streams"] == ["video.mp4"]


def test_build_submit_payload_matches_ios_shape():
    metadata = {
        "device": {"model": "iPhone16,1"},
        "durationMs": 1234,
        "gps": {"lat": -33.1, "lon": 151.2, "accuracyM": 3.4},
        "streams": ["video.mp4", "imu.jsonl"],
    }

    payload = build_submit_payload(
        metadata,
        recording_id="rec",
        task_id="task",
        size_bytes=99,
    )

    assert payload == {
        "recording_id": "rec",
        "task_id": "task",
        "device_model": "iPhone16,1",
        "duration_ms": 1234,
        "size_bytes": 99,
        "gps_lat": -33.1,
        "gps_lon": 151.2,
        "gps_accuracy_m": 3.4,
        "storage_path": "rec/",
        "streams": ["video.mp4", "imu.jsonl"],
    }
```

- [ ] **Step 2: Implement the E2E CLI**

Create `backend/backend/tools/__init__.py` with:

```python
"""Local backend tools."""
```

Create `backend/backend/tools/e2e_upload_bundle.py` with:

```python
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from backend.supabase_api import SupabaseApi, SupabaseConfig


CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".bin": "application/octet-stream",
}


def patch_metadata(metadata: dict[str, Any], *, recording_id: str, task_id: str) -> dict[str, Any]:
    patched = dict(metadata)
    patched["recordingId"] = recording_id
    patched["bountyId"] = task_id
    return patched


def build_submit_payload(
    metadata: dict[str, Any],
    *,
    recording_id: str,
    task_id: str,
    size_bytes: int,
) -> dict[str, Any]:
    gps = metadata.get("gps") or {}
    device = metadata.get("device") or {}
    return {
        "recording_id": recording_id,
        "task_id": task_id,
        "device_model": device.get("model"),
        "duration_ms": metadata.get("durationMs"),
        "size_bytes": size_bytes,
        "gps_lat": gps.get("lat"),
        "gps_lon": gps.get("lon"),
        "gps_accuracy_m": gps.get("accuracyM"),
        "storage_path": f"{recording_id}/",
        "streams": metadata.get("streams") or [],
    }


def bundle_size(bundle: Path) -> int:
    return sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def password_token(config: SupabaseConfig, email: str, password: str) -> str:
    res = httpx.post(
        f"{config.url}/auth/v1/token?grant_type=password",
        headers={"apikey": config.key, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=60,
    )
    res.raise_for_status()
    return res.json()["access_token"]


def upload_bundle(api: SupabaseApi, bundle: Path, *, recording_id: str, task_id: str) -> dict[str, Any]:
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    patched_metadata = patch_metadata(metadata, recording_id=recording_id, task_id=task_id)

    for file_path in sorted(bundle.iterdir()):
        if not file_path.is_file():
            continue
        object_path = f"{recording_id}/{file_path.name}"
        if file_path.name == "metadata.json":
            api.upload_json("recordings", object_path, patched_metadata)
        else:
            api.upload_file("recordings", object_path, file_path, content_type(file_path))

    return build_submit_payload(
        patched_metadata,
        recording_id=recording_id,
        task_id=task_id,
        size_bytes=bundle_size(bundle),
    )


def submit_recording(config: SupabaseConfig, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    res = httpx.post(
        f"{config.url}/functions/v1/submit-recording",
        headers={
            "apikey": config.key,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    res.raise_for_status()
    return res.json()


def poll_score(api: SupabaseApi, recording_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = api.select_one(
            "recordings",
            "id=eq."
            + recording_id
            + "&select=id,is_scoring,summary,success,success_reasoning,score,score_reasoning,status",
        )
        if not row.get("is_scoring"):
            return row
        print("score pending...")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for score for {recording_id}")


def poll_all_jobs(api: SupabaseApi, recording_id: str, timeout_s: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    terminal = {"succeeded", "failed"}
    while time.monotonic() < deadline:
        res = api.client.get(
            f"{api.config.url}/rest/v1/recording_analysis_jobs"
            f"?recording_id=eq.{recording_id}"
            "&select=kind,status,artifact_path,error"
            "&order=kind.asc",
            headers=api.rest_headers(),
        )
        rows = api._json(res)
        if len(rows) == 5 and all(row["status"] in terminal for row in rows):
            for row in rows:
                artifact_path = row.get("artifact_path")
                if row["status"] == "succeeded" and artifact_path:
                    api.download_bytes("recordings", artifact_path)
            return rows
        print("analysis jobs pending...")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for analysis jobs for {recording_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--recording-id", default="")
    parser.add_argument("--wait", choices=["none", "score", "all"], default="score")
    parser.add_argument("--timeout-s", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording_id = (args.recording_id or str(uuid.uuid4())).lower()
    bundle = args.bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise SystemExit(f"Bundle directory does not exist: {bundle}")

    config = SupabaseConfig.from_anon_env()
    token = os.environ.get("SUPABASE_ACCESS_TOKEN")
    if not token:
        email = os.environ.get("TEST_COLLECTOR_EMAIL")
        password = os.environ.get("TEST_COLLECTOR_PASSWORD")
        if not email or not password:
            raise SystemExit("Set SUPABASE_ACCESS_TOKEN or TEST_COLLECTOR_EMAIL/TEST_COLLECTOR_PASSWORD")
        token = password_token(config, email, password)

    api = SupabaseApi(config, bearer_token=token)
    try:
        payload = upload_bundle(api, bundle, recording_id=recording_id, task_id=args.task_id)
        response = submit_recording(config, token, payload)
        print(json.dumps({"submit": response, "recording_id": recording_id}, indent=2))
        if args.wait in {"score", "all"}:
            score = poll_score(api, recording_id, args.timeout_s)
            print(json.dumps({"score": score}, indent=2))
        if args.wait == "all":
            jobs = poll_all_jobs(api, recording_id, args.timeout_s)
            print(json.dumps({"analysis_jobs": jobs}, indent=2))
    finally:
        api.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

Run:

```bash
cd backend
uv run --python 3.12 pytest tests/test_e2e_bundle.py -q
```

Expected: PASS.

- [ ] **Step 4: Run dry import check**

Run:

```bash
cd backend
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle --help
```

Expected: command help prints without import errors.

- [ ] **Step 5: Commit**

Run:

```bash
git add backend/backend/tools backend/tests/test_e2e_bundle.py
git commit -m "Add E2E recording upload tool"
```

## Task 7: Minimal Web Status View And Task Object Tags

**Files:**
- Modify: `web/src/app/lab/tasks/new/page.tsx`
- Modify: `web/src/app/lab/tasks/new/create-task-form.tsx`
- Modify: `web/src/app/lab/tasks/[id]/page.tsx`
- Modify: `web/src/app/lab/tasks/[id]/submissions-live.tsx`

- [ ] **Step 1: Persist object tags from new task form**

In `web/src/app/lab/tasks/new/page.tsx`, parse `objects`:

```ts
const objects = String(formData.get('objects') ?? '')
  .split(',')
  .map((item) => item.trim().toLowerCase())
  .filter(Boolean)
```

Add `objects` to the `tasks` insert payload:

```ts
objects,
```

- [ ] **Step 2: Add simple object tags input**

In `web/src/app/lab/tasks/new/create-task-form.tsx`, add state near the other state hooks:

```ts
const [objects, setObjects] = useState('')
```

Add this section after the requirements section:

```tsx
<section className="surface-panel p-5 sm:p-6">
  <div className="mb-4">
    <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Objects to segment</h2>
    <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
      Comma-separated object tags for SAM. Human hand is always included automatically.
    </p>
  </div>
  <input
    name="objects"
    value={objects}
    onChange={(event) => setObjects(event.target.value)}
    placeholder="cup, can, spoon"
    className="input-dark text-sm"
  />
</section>
```

- [ ] **Step 3: Load recordings and analysis jobs in the lab task page**

In `web/src/app/lab/tasks/[id]/page.tsx`, after fetching `rawSubmissions`, derive recording IDs from `metadata.recording_id`:

```ts
const recordingIds = (rawSubmissions ?? [])
  .map((submission) => (submission.metadata as { recording_id?: string } | null)?.recording_id)
  .filter((value): value is string => Boolean(value))
```

Fetch recordings and jobs:

```ts
const { data: recordings } = recordingIds.length
  ? await supabase
      .from('recordings')
      .select('id, status, is_scoring, summary, success, success_reasoning, score, score_reasoning, detected_objects, analysis_artifacts')
      .in('id', recordingIds)
  : { data: [] }

const { data: analysisJobs } = recordingIds.length
  ? await supabase
      .from('recording_analysis_jobs')
      .select('recording_id, kind, status, artifact_path, summary, error, started_at, finished_at')
      .in('recording_id', recordingIds)
  : { data: [] }
```

When building `submissions`, attach matching recording and jobs:

```ts
const recordingById = new Map((recordings ?? []).map((recording) => [recording.id, recording]))
const jobsByRecordingId = new Map<string, typeof analysisJobs>()
for (const job of analysisJobs ?? []) {
  const list = jobsByRecordingId.get(job.recording_id) ?? []
  list.push(job)
  jobsByRecordingId.set(job.recording_id, list)
}

const submissions = await Promise.all(
  (rawSubmissions ?? []).map(async (s) => {
    const recordingId = (s.metadata as { recording_id?: string } | null)?.recording_id ?? null
    const { data } = await supabase.storage
      .from('recordings')
      .createSignedUrl(s.storage_path.replace(/\/$/, '') + '/video.mp4', 3600)
    return {
      ...s,
      signedUrl: data?.signedUrl ?? null,
      recording: recordingId ? recordingById.get(recordingId) ?? null : null,
      analysisJobs: recordingId ? jobsByRecordingId.get(recordingId) ?? [] : [],
    }
  })
)
```

- [ ] **Step 4: Render compact analysis status in submission cards**

In `web/src/app/lab/tasks/[id]/submissions-live.tsx`, extend `Submission`:

```ts
type AnalysisJob = {
  recording_id: string
  kind: string
  status: 'pending' | 'running' | 'succeeded' | 'failed'
  artifact_path: string | null
  summary: Record<string, unknown> | null
  error: string | null
  started_at: string | null
  finished_at: string | null
}

type RecordingAnalysis = {
  id: string
  status: string
  is_scoring: boolean | null
  summary: string | null
  success: boolean | null
  success_reasoning: string | null
  score: number | null
  score_reasoning: string | null
  detected_objects: unknown
  analysis_artifacts: Record<string, string> | null
}
```

Add to `Submission`:

```ts
recording: RecordingAnalysis | null
analysisJobs: AnalysisJob[]
```

Inside `SubmissionCard`, render after metadata:

```tsx
{submission.recording && (
  <div className="mt-4 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-3">
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Analysis</span>
      <span className="rounded-full border border-[var(--border)] px-2 py-0.5 text-xs text-[var(--foreground-secondary)]">
        {submission.recording.status}
      </span>
      {submission.recording.is_scoring ? (
        <span className="rounded-full bg-[rgba(216,163,71,0.16)] px-2 py-0.5 text-xs text-[#f0cb7c]">Scoring</span>
      ) : (
        <span className="rounded-full bg-[rgba(47,158,68,0.16)] px-2 py-0.5 text-xs text-[#99ddaa]">Score ready</span>
      )}
    </div>

    {submission.recording.score !== null && (
      <div className="mt-3 grid gap-2 sm:grid-cols-[90px_minmax(0,1fr)]">
        <div className="text-2xl font-bold text-white">{submission.recording.score}/10</div>
        <div>
          {submission.recording.summary && (
            <p className="text-sm text-white">{submission.recording.summary}</p>
          )}
          {submission.recording.score_reasoning && (
            <p className="mt-1 text-xs text-[var(--foreground-secondary)]">{submission.recording.score_reasoning}</p>
          )}
        </div>
      </div>
    )}

    {submission.analysisJobs.length > 0 && (
      <div className="mt-3 flex flex-wrap gap-2">
        {submission.analysisJobs.map((job) => (
          <span
            key={job.kind}
            className="rounded-full border border-[var(--border)] px-2 py-0.5 text-xs text-[var(--foreground-secondary)]"
            title={job.error ?? job.artifact_path ?? undefined}
          >
            {job.kind}: {job.status}
          </span>
        ))}
      </div>
    )}
  </div>
)}
```

- [ ] **Step 5: Run web checks**

Run:

```bash
cd web
npm run lint
npm run build
```

Expected: both commands pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add web/src/app/lab/tasks/new/page.tsx web/src/app/lab/tasks/new/create-task-form.tsx 'web/src/app/lab/tasks/[id]/page.tsx' 'web/src/app/lab/tasks/[id]/submissions-live.tsx'
git commit -m "Show recording analysis status in lab submissions"
```

## Task 8: Full E2E Verification And Demo Runbook

**Files:**
- Modify: `backend/README.md`

- [ ] **Step 1: Configure secrets**

Set Supabase Edge Function secrets:

```bash
supabase secrets set \
  MODAL_ANALYSIS_URL="https://<modal-submit-analysis-url>" \
  MODAL_ANALYSIS_SECRET="<shared-secret>" \
  --project-ref coapgtbwmzxkfewzncxu
```

Set Modal secret:

```bash
cd backend
uv run --python 3.12 modal secret create copilot-hackathon-backend-secrets \
  SUPABASE_URL="https://coapgtbwmzxkfewzncxu.supabase.co" \
  SUPABASE_SERVICE_ROLE_KEY="<service-role-key>" \
  SUPABASE_ANON_KEY="<anon-key>" \
  GEMINI_API_KEY="<gemini-key>" \
  MODAL_ANALYSIS_SECRET="<shared-secret>"
```

Expected: both commands complete without printing secret values back into logs.

- [ ] **Step 2: Run backend unit tests**

Run:

```bash
cd backend
uv run --python 3.12 pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run the E2E score path**

Run:

```bash
cd backend
TEST_COLLECTOR_EMAIL="<collector-email>" \
TEST_COLLECTOR_PASSWORD="<collector-password>" \
SUPABASE_URL="https://coapgtbwmzxkfewzncxu.supabase.co" \
SUPABASE_ANON_KEY="<anon-key>" \
uv run --python 3.12 python -m backend.tools.e2e_upload_bundle \
  --bundle ../playground/data/iphone-data-2 \
  --task-id 106760b6-43ec-41bd-b6f6-340b00db1d58 \
  --wait score
```

Expected:

- upload completes
- `submit-recording` returns `ok: true`
- `analysis_started` is `true`
- polling eventually prints `is_scoring: false`
- score fields are populated or the Gemini job row contains a clear failure

- [ ] **Step 4: Inspect artifacts and jobs**

Run a SQL query:

```sql
select kind, status, artifact_path, error
from public.recording_analysis_jobs
where recording_id = '<recording-id-from-e2e>'
order by kind;
```

Expected: five rows exist. For successful jobs, `artifact_path` points under `<recording-id>/analysis/`.

- [ ] **Step 5: Run web checks**

Run:

```bash
cd web
npm run lint
npm run build
```

Expected: PASS.

- [ ] **Step 6: Update backend README with the verified command**

Append the exact successful E2E command and the expected output fields to `backend/README.md`. Do not include secret values.

- [ ] **Step 7: Commit**

Run:

```bash
git add backend/README.md
git commit -m "Document backend E2E verification"
```

## Plan Self-Review

Spec coverage:

- Upload path remains iOS Storage plus `submit-recording`: Task 3 and Task 6.
- Gemini score returns early through `is_scoring`: Task 3, Task 4, Task 5, Task 6.
- MediaPipe, YOLO, SAM, temporal artifacts: Task 5.
- `tasks.objects` for SAM prompts and automatic `human hand`: Task 1 and Task 7.
- Durable per-analyzer rows: Task 2, Task 3, Task 5.
- Minimal web view: Task 7.
- E2E fixture path using `playground/data/iphone-data-2`: Task 6 and Task 8.
- TwelveLabs/search functions untouched: File Structure and Scope Check.

Type consistency:

- Analyzer kinds match `AnalysisKind`, migration checks, Edge Function rows, artifact path keys, and web job display.
- Scoring fields match existing Supabase fields and `GeminiEvaluation`.
- E2E payload uses the same request fields as the iOS upload service.

Verification gates:

- Backend unit tests after Tasks 1, 4, 5, and 6.
- Supabase schema verification after Task 2.
- Deno/Supabase Edge check after Task 3.
- Web lint/build after Task 7.
- Live E2E after Task 8.
