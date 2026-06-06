"""Feature 1 — Semantic demo search.

After all analyzers complete, aggregate their outputs into one natural-language
document, embed it, and upsert into Pinecone namespace "recordings" (one vector
per recording). Labs can then semantically search their dataset from the Studio.

This is a fire-and-forget post-processing step: it must never raise into the
pipeline.
"""

from __future__ import annotations

from typing import Any

from backend.pinecone_client import (
    RECORDINGS_NAMESPACE,
    embed_document,
    get_pinecone_index,
)


def _detected_object_names(yolo_detections: dict[str, Any]) -> list[str]:
    """Unique detected object class names, tolerant of both the real YOLO schema
    (frames[*].instances[*].class_name) and the simpler {detections:[...]} shape.
    """
    names: set[str] = set()
    for frame in yolo_detections.get("frames", []) or []:
        instances = frame.get("instances") or frame.get("detections") or []
        for det in instances:
            name = det.get("class_name")
            if name:
                names.add(str(name))
    return sorted(names)


def _action_phases(temporal_actions: dict[str, Any]) -> list[str]:
    """Unique action labels, tolerant of {phases:[{action}]} and the real
    temporal-action {segments:[{label}]} shape."""
    items = temporal_actions.get("phases") or temporal_actions.get("segments") or []
    phases: set[str] = set()
    for item in items:
        label = item.get("action") or item.get("label") or item.get("phase")
        if label:
            phases.add(str(label))
    return sorted(phases)


def _passed(gemini_eval: dict[str, Any]) -> bool:
    # Real gemini-eval.json uses "success"; the spec/tests use "passed".
    if "passed" in gemini_eval:
        return bool(gemini_eval.get("passed"))
    return bool(gemini_eval.get("success", False))


def build_recording_document(
    recording_id: str,
    task_title: str,
    task_description: str,
    objects_to_detect: list[str],
    lab_id: str,
    gemini_eval: dict,
    yolo_detections: dict,
    temporal_actions: dict,
    temporal_annotations: dict | None,
) -> str:
    """Build one natural-language document capturing everything meaningful about
    this recording for semantic retrieval."""
    summary = gemini_eval.get("summary", "")
    score = gemini_eval.get("score", 0)
    passed = _passed(gemini_eval)

    detected_objects = _detected_object_names(yolo_detections)
    phases = _action_phases(temporal_actions)

    scene_summary = ""
    searchable_tags: list[str] = []
    key_moment_descriptions: list[str] = []
    if temporal_annotations:
        scene_summary = temporal_annotations.get("scene_summary", "")
        searchable_tags = temporal_annotations.get("searchable_tags", []) or []
        for m in temporal_annotations.get("key_moments", []) or []:
            ts = float(m.get("timestamp_seconds", 0.0) or 0.0)
            key_moment_descriptions.append(
                f"{ts:.1f}s: {m.get('action', '')} ({m.get('phase', '')})"
            )

    document = (
        f"Task: {task_title}. {task_description}. "
        f"Required objects: {', '.join(objects_to_detect)}. "
        f"Detected objects: {', '.join(detected_objects)}. "
        f"Action phases: {', '.join(phases)}. "
        f"Scene: {summary}. {scene_summary}. "
        f"Key moments: {' | '.join(key_moment_descriptions[:10])}. "
        f"Tags: {', '.join(searchable_tags)}. "
        f"Completed: {passed}. Quality score: {score}/10."
    )
    return document


def index_recording(
    recording_id: str,
    task_title: str,
    task_description: str,
    objects_to_detect: list[str],
    lab_id: str,
    collector_id: str,
    gemini_eval: dict,
    yolo_detections: dict,
    temporal_actions: dict,
    temporal_annotations: dict | None,
    gemini_score: float,
    task_id: str,
) -> None:
    """Build the document, embed it, and upsert to Pinecone namespace
    "recordings". Wrapped in try/except — never crashes the pipeline."""
    try:
        document = build_recording_document(
            recording_id,
            task_title,
            task_description,
            objects_to_detect,
            lab_id,
            gemini_eval,
            yolo_detections,
            temporal_actions,
            temporal_annotations,
        )
        vector = embed_document(document)
        index = get_pinecone_index()
        index.upsert(
            vectors=[
                {
                    "id": recording_id,
                    "values": vector,
                    "metadata": {
                        "recording_id": recording_id,
                        "task_id": task_id or "",
                        "task_title": task_title or "",
                        "lab_id": lab_id or "",
                        "collector_id": collector_id or "",
                        "objects": objects_to_detect or [],
                        "detected_objects": _detected_object_names(yolo_detections),
                        "gemini_score": float(gemini_score or 0.0),
                        "passed": _passed(gemini_eval),
                        "embedding_document": document[:800],
                    },
                }
            ],
            namespace=RECORDINGS_NAMESPACE,
        )
        print(f"[recording_indexer] Upserted recording {recording_id} to Pinecone")
    except Exception as e:  # noqa: BLE001 - fire-and-forget, never fail the pipeline
        print(f"[recording_indexer] ERROR indexing recording {recording_id}: {e}")
