from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from backend.contracts import SCORING_ANALYSIS_KINDS, UNCONDITIONAL_ANALYSIS_KINDS
from backend.supabase_api import SupabaseApi, SupabaseConfig


CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".bin": "application/octet-stream",
}


def resource_intensive_analysis_enabled() -> bool:
    return os.environ.get("COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def patch_metadata(
    metadata: dict[str, Any], *, recording_id: str, task_id: str
) -> dict[str, Any]:
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
    return str(res.json()["access_token"])


def upload_bundle(
    api: SupabaseApi, bundle: Path, *, recording_id: str, task_id: str
) -> dict[str, Any]:
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    patched_metadata = patch_metadata(
        metadata, recording_id=recording_id, task_id=task_id
    )

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


def submit_recording(
    config: SupabaseConfig, token: str, payload: dict[str, Any]
) -> dict[str, Any]:
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
    try:
        res.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"submit-recording failed {res.status_code}: {res.text}"
        ) from exc
    return res.json()


def normalize_recording_id(recording_id: str) -> str:
    try:
        return str(uuid.UUID(recording_id))
    except ValueError as exc:
        raise ValueError(f"recording_id must be a UUID: {recording_id}") from exc


def ensure_analysis_started(response: dict[str, Any]) -> None:
    if response.get("analysis_started") is False:
        error = response.get("analysis_error") or "analysis_started=false"
        raise RuntimeError(f"Analysis did not start: {error}")


def _valid_score_row(row: dict[str, Any]) -> bool:
    return (
        row.get("score") is not None
        and row.get("success") is not None
        and bool(row.get("summary"))
    )


def _gemini_job(api: SupabaseApi, recording_id: str) -> dict[str, Any] | None:
    res = api.client.get(
        f"{api.config.url}/rest/v1/recording_analysis_jobs"
        f"?recording_id=eq.{recording_id}"
        "&kind=eq.gemini_eval"
        "&select=kind,status,artifact_path,error",
        headers=api.rest_headers(),
    )
    rows = api._json(res) or []
    return rows[0] if rows else None


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
            if not _valid_score_row(row):
                gemini_job = _gemini_job(api, recording_id)
                raise RuntimeError(
                    "Scoring finished without complete Gemini fields for "
                    f"{recording_id}: row={json.dumps(row, sort_keys=True, default=str)} "
                    f"gemini_job={json.dumps(gemini_job, sort_keys=True, default=str)}"
                )
            return row

        print("score pending...")
        time.sleep(5)

    raise TimeoutError(f"Timed out waiting for score for {recording_id}")


def poll_all_jobs(
    api: SupabaseApi, recording_id: str, timeout_s: int
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    terminal = {"succeeded", "failed"}
    required_kinds = set(
        UNCONDITIONAL_ANALYSIS_KINDS
        if resource_intensive_analysis_enabled()
        else SCORING_ANALYSIS_KINDS
    )
    while time.monotonic() < deadline:
        res = api.client.get(
            f"{api.config.url}/rest/v1/recording_analysis_jobs"
            f"?recording_id=eq.{recording_id}"
            "&select=kind,status,artifact_path,error"
            "&order=kind.asc",
            headers=api.rest_headers(),
        )
        rows = api._json(res)
        kinds_seen = {row["kind"] for row in rows}
        # All unconditional kinds present and every present row is terminal.
        # gaussian_splat is optional (depth-gated) so we don't require it.
        if (
            required_kinds.issubset(kinds_seen)
            and all(row["status"] in terminal for row in rows)
        ):
            failed_rows = [row for row in rows if row["status"] == "failed"]
            if failed_rows:
                summary = "; ".join(
                    f"{row.get('kind')}: {row.get('error') or 'unknown error'}"
                    for row in failed_rows
                )
                raise RuntimeError(
                    f"Analysis jobs failed for {recording_id}: {summary}"
                )

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
    recording_id = normalize_recording_id(args.recording_id or str(uuid.uuid4()))
    bundle = args.bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise SystemExit(f"Bundle directory does not exist: {bundle}")

    config = SupabaseConfig.from_anon_env()
    token = os.environ.get("SUPABASE_ACCESS_TOKEN")
    if not token:
        email = os.environ.get("TEST_COLLECTOR_EMAIL")
        password = os.environ.get("TEST_COLLECTOR_PASSWORD")
        if not email or not password:
            raise SystemExit(
                "Set SUPABASE_ACCESS_TOKEN or TEST_COLLECTOR_EMAIL/TEST_COLLECTOR_PASSWORD"
            )
        token = password_token(config, email, password)

    api = SupabaseApi(config, bearer_token=token)
    try:
        payload = upload_bundle(
            api, bundle, recording_id=recording_id, task_id=args.task_id
        )
        response = submit_recording(config, token, payload)
        print(json.dumps({"submit": response, "recording_id": recording_id}, indent=2))
        try:
            ensure_analysis_started(response)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

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
