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
