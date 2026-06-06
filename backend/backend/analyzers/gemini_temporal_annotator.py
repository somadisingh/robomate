"""Gemini Temporal Annotator.

Watches the full recording video and produces structured timestamped
annotations (key moments, object visibility, failure moments, searchable tags).

Output artifact: recordings/<id>/analysis/gemini-temporal-annotations.json
The annotation is also embedded and upserted into Pinecone so labs can run
semantic search over recordings later.

This module uses the same `google-genai` SDK as analyzers/gemini_eval.py (not the
legacy `google-generativeai` package) so the project keeps a single Gemini SDK.
All external clients (Gemini, Pinecone) are dependency-injectable so the logic is
unit-testable without network access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.analyzers.gemini_eval import wait_for_uploaded_file

# Default Gemini model for video understanding. Overridable via env so it tracks
# whatever the rest of the pipeline uses.
DEFAULT_ANNOTATION_MODEL = "gemini-3.5-flash"
# gemini-embedding-001 supports configurable output dimensions; we request 768 to
# match the Pinecone index dimension via Matryoshka (MRL) truncation.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


ANNOTATION_PROMPT = """
You are analyzing a robot learning demonstration video. A human is performing a physical manipulation task.

Task title: {task_title}
Task description: {task_description}
Objects that should appear: {objects}

Watch the entire video and return ONLY a valid JSON object with this exact schema - no markdown, no explanation:

{{
  "duration_seconds": <float, total video duration>,
  "task_completed": <true|false>,
  "completion_confidence": <0.0 to 1.0>,
  "scene_summary": "<2-3 sentence description of what happened overall>",
  "key_moments": [
    {{
      "timestamp_seconds": <float>,
      "action": "<short verb phrase, e.g. 'reaches for bottle'>",
      "objects_involved": ["<object1>", "<object2>"],
      "hand": "<left|right|both|none>",
      "phase": "<approach|grasp|transport|place|release|idle>"
    }}
  ],
  "object_visibility": {{
    "<object_name>": {{
      "first_visible_at": <float seconds or null>,
      "last_visible_at": <float seconds or null>,
      "total_visible_seconds": <float>
    }}
  }},
  "failure_moments": [
    {{
      "timestamp_seconds": <float>,
      "description": "<what went wrong>"
    }}
  ],
  "searchable_tags": ["<tag1>", "<tag2>"]
}}

Return ONLY the JSON. No other text.
"""


def build_annotation_prompt(
    task_title: str, task_description: str, objects_to_detect: list[str]
) -> str:
    return ANNOTATION_PROMPT.format(
        task_title=task_title,
        task_description=task_description,
        objects=", ".join(objects_to_detect),
    )


def pinecone_enabled() -> bool:
    """True only when both Pinecone env vars are present.

    Lets the pipeline run end-to-end (and write the JSON artifact) before the
    Pinecone credentials have been added to the Modal secret.
    """
    return bool(
        os.environ.get("PINECONE_API_KEY") and os.environ.get("PINECONE_INDEX_NAME")
    )


def run_temporal_annotation(
    *,
    recording_id: str,
    video_path: str | Path,
    task_title: str,
    task_description: str,
    objects_to_detect: list[str],
    client: Any | None = None,
    model: str | None = None,
    upload_timeout_s: float = 120,
    poll_interval_s: float = 5,
) -> dict[str, Any]:
    """Upload the video to the Gemini Files API, run the annotation prompt, and
    return the parsed JSON annotations."""
    resolved_video = Path(video_path).expanduser().resolve()
    if not resolved_video.is_file():
        raise FileNotFoundError(f"video file does not exist: {resolved_video}")

    active_model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_ANNOTATION_MODEL
    active_client = client or _make_client()

    print(f"[temporal_annotator] uploading video for recording {recording_id}")
    uploaded_file = active_client.files.upload(file=str(resolved_video))
    try:
        uploaded_file = wait_for_uploaded_file(
            active_client,
            uploaded_file,
            timeout_s=upload_timeout_s,
            poll_interval_s=poll_interval_s,
        )
        prompt = build_annotation_prompt(
            task_title, task_description, objects_to_detect
        )
        response = active_client.models.generate_content(
            model=active_model,
            contents=[uploaded_file, prompt],
            config=_annotation_config(),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty annotation response")
        return _parse_annotation_json(text)
    finally:
        _delete_uploaded_file(active_client, uploaded_file)


def _parse_annotation_json(text: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating ```json ... ``` markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        # Take the content between the first pair of fences.
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def build_pinecone_embedding(
    *,
    recording_id: str,
    task_title: str,
    task_description: str,
    objects_to_detect: list[str],
    annotations: dict[str, Any],
    gemini_score: float | None,
    client: Any | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Build a natural-language document from the analysis outputs, embed it with
    Gemini's embedding model, and return a Pinecone upsert payload."""
    key_moments_text = " | ".join(
        f"{float(m.get('timestamp_seconds', 0.0)):.1f}s: {m.get('action', '')} "
        f"({m.get('phase', '')})"
        for m in annotations.get("key_moments", [])
    )
    tags_text = ", ".join(annotations.get("searchable_tags", []))
    objects_text = ", ".join(objects_to_detect)

    embedding_document = (
        f"Task: {task_title}. {task_description}. "
        f"Objects: {objects_text}. "
        f"Scene: {annotations.get('scene_summary', '')}. "
        f"Key moments: {key_moments_text}. "
        f"Tags: {tags_text}. "
        f"Completed: {annotations.get('task_completed')}. "
        f"Gemini quality score: {gemini_score}."
    )

    active_client = client or _make_client()
    active_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    vector = _embed_document(active_client, active_model, embedding_document)

    return {
        "id": recording_id,
        "values": vector,
        "metadata": {
            "recording_id": recording_id,
            "task_title": task_title,
            "objects": objects_to_detect,
            "task_completed": bool(annotations.get("task_completed", False)),
            "completion_confidence": float(
                annotations.get("completion_confidence", 0.0) or 0.0
            ),
            "gemini_score": float(gemini_score or 0.0),
            "scene_summary": annotations.get("scene_summary", ""),
            "searchable_tags": annotations.get("searchable_tags", []),
            "key_moment_count": len(annotations.get("key_moments", [])),
            "failure_count": len(annotations.get("failure_moments", [])),
            # Pinecone metadata values must stay small; truncate the document.
            "embedding_document": embedding_document[:1000],
        },
    }


def upsert_to_pinecone(
    pinecone_payload: dict[str, Any],
    *,
    client: Any | None = None,
    index_name: str | None = None,
) -> None:
    """Upsert a single vector into Pinecone, creating the index on first use."""
    active_index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")
    if not active_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is required to upsert to Pinecone")

    pc = client or _make_pinecone_client()

    existing = [getattr(i, "name", None) or i.get("name") for i in pc.list_indexes()]
    if active_index_name not in existing:
        from pinecone import ServerlessSpec

        pc.create_index(
            name=active_index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(active_index_name)
    index.upsert(vectors=[pinecone_payload])
    print(
        f"[temporal_annotator] upserted recording {pinecone_payload['id']} to Pinecone"
    )


# --------------------------------------------------------------------------- #
# Internal client factories / extraction helpers
# --------------------------------------------------------------------------- #


def _make_client() -> Any:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment.")
    return genai.Client(api_key=api_key)


def _make_pinecone_client() -> Any:
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in the environment.")
    return Pinecone(api_key=api_key)


def _annotation_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )


def _embed_document(client: Any, model: str, document: str) -> list[float]:
    config: Any = None
    try:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION,
        )
    except Exception:
        config = None

    if config is not None:
        result = client.models.embed_content(
            model=model, contents=document, config=config
        )
    else:
        result = client.models.embed_content(model=model, contents=document)
    return _extract_embedding(result)


def _extract_embedding(result: Any) -> list[float]:
    """Pull the float vector out of the various shapes embed_content can return."""
    # New google-genai: result.embeddings -> [ContentEmbedding(values=[...])]
    embeddings = getattr(result, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return list(values)
        if isinstance(first, dict) and "values" in first:
            return list(first["values"])
    # Singular .embedding with .values
    embedding = getattr(result, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return list(values)
    # Dict forms (mocks / legacy)
    if isinstance(result, dict):
        if "embedding" in result:
            emb = result["embedding"]
            return list(emb["values"]) if isinstance(emb, dict) else list(emb)
        if "embeddings" in result and result["embeddings"]:
            emb = result["embeddings"][0]
            return list(emb["values"]) if isinstance(emb, dict) else list(emb)
    raise RuntimeError("Could not extract embedding vector from embed_content result")


def _delete_uploaded_file(client: Any, uploaded_file: Any) -> None:
    name = getattr(uploaded_file, "name", None)
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception:
        pass
