from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.artifacts import analysis_artifact_paths
from backend.analyzers.gemini_eval import evaluate_video_file
from backend.contracts import (
    AnalysisKind,
    AnalysisRequest,
    GeminiEvaluation,
    RecordingRecord,
    RESOURCE_INTENSIVE_ANALYSIS_KINDS,
    TaskRecord,
)
from backend.supabase_api import SupabaseApi


RECORDINGS_BUCKET = "recordings"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_job(
    api: SupabaseApi,
    recording_id: str,
    kind: AnalysisKind,
    status: str,
    **fields: Any,
) -> None:
    payload = {"status": status, **fields}
    api.patch_rows(
        "recording_analysis_jobs",
        f"recording_id=eq.{recording_id}&kind=eq.{kind}",
        payload,
    )


def prune_resource_intensive_jobs(api: SupabaseApi, recording_id: str) -> None:
    kinds = ",".join(RESOURCE_INTENSIVE_ANALYSIS_KINDS)
    api.delete_rows(
        "recording_analysis_jobs",
        f"recording_id=eq.{recording_id}&kind=in.({kinds})",
    )


def apply_gemini_result(
    api: SupabaseApi,
    recording_id: str,
    result: GeminiEvaluation,
) -> None:
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


def fetch_context(
    api: SupabaseApi,
    request: AnalysisRequest,
) -> tuple[TaskRecord, RecordingRecord, bytes]:
    task = TaskRecord.model_validate(
        api.select_one("tasks", f"id=eq.{request.task_id}&select=*")
    )
    recording = RecordingRecord.model_validate(
        api.select_one("recordings", f"id=eq.{request.recording_id}&select=*")
    )
    video_path = _recording_video_path(request.storage_path)
    video_bytes = api.download_bytes(RECORDINGS_BUCKET, video_path)
    return task, recording, video_bytes


def run_gemini(
    api: SupabaseApi,
    request: AnalysisRequest,
    task: TaskRecord,
    video_bytes: bytes,
) -> dict[str, Any]:
    paths = analysis_artifact_paths(request.recording_id)
    mark_job(
        api,
        request.recording_id,
        "gemini_eval",
        "running",
        error=None,
        started_at=utc_now(),
        finished_at=None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "video.mp4"
        video_path.write_bytes(video_bytes)
        result = evaluate_video_file(
            video_path=video_path,
            task_description=task.description or task.title,
        )

    payload = result.model_dump()
    api.upload_json(RECORDINGS_BUCKET, paths["gemini_eval"], payload)
    apply_gemini_result(api, request.recording_id, result)
    mark_job(
        api,
        request.recording_id,
        "gemini_eval",
        "succeeded",
        artifact_path=paths["gemini_eval"],
        summary=payload,
        error=None,
        finished_at=utc_now(),
    )
    return payload


def run_remote_analyzer(
    api: SupabaseApi,
    request: AnalysisRequest,
    kind: AnalysisKind,
    artifact_payload: dict[str, Any],
    db_summary: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> str:
    paths = analysis_artifact_paths(request.recording_id)
    api.upload_json(RECORDINGS_BUCKET, paths[kind], artifact_payload)
    mark_job(
        api,
        request.recording_id,
        kind,
        "succeeded",
        artifact_path=paths[kind],
        summary=db_summary,
        error=None,
        finished_at=utc_now(),
    )
    return paths[kind]


def finalize_multi_artifact_job(
    api: SupabaseApi,
    request: AnalysisRequest,
    kind: AnalysisKind,
    artifact_path: str,
    db_summary: dict[str, Any] | None = None,
) -> str:
    """Mark a multi-file job as succeeded. The Modal function uploads each
    file itself; the orchestrator only records the artifact_path that the web
    studio should fetch (typically a manifest.json) and the row-level summary.
    """
    mark_job(
        api,
        request.recording_id,
        kind,
        "succeeded",
        artifact_path=artifact_path,
        summary=db_summary,
        error=None,
        finished_at=utc_now(),
    )
    return artifact_path


# Minimum number of depth frames a recording needs before we attempt to train
# a gaussian splat. Below this the dataset is too sparse to be worth the GPU.
MIN_DEPTH_FRAMES_FOR_SPLAT = 30


def gaussian_splat_preflight(recording: RecordingRecord) -> bool:
    """Return True iff a recording has enough LiDAR depth data to train on.

    We require the depth grid dimensions to be set (added in the
    20260524170000 migration) and at least ``MIN_DEPTH_FRAMES_FOR_SPLAT``
    depth frames.
    """
    extras = recording.model_dump()
    depth_width = extras.get("depth_width")
    depth_height = extras.get("depth_height")
    depth_frame_count = extras.get("depth_frame_count")
    if not (depth_width and depth_height and depth_frame_count):
        return False
    return int(depth_frame_count) >= MIN_DEPTH_FRAMES_FOR_SPLAT


def update_final_status(api: SupabaseApi, recording_id: str) -> str:
    response = api.client.get(
        f"{api.config.url}/rest/v1/recording_analysis_jobs"
        f"?recording_id=eq.{recording_id}&select=status",
        headers=api.rest_headers(),
    )
    rows = api._json(response)
    statuses = [row["status"] for row in rows]
    status = final_recording_status(statuses)
    api.patch_rows("recordings", f"id=eq.{recording_id}", {"status": status})
    return status


def _recording_video_path(storage_path: str) -> str:
    return f"{storage_path.strip().strip('/')}/video.mp4"
