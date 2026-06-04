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
    "gaussian_splat": "gaussian_splat/manifest.json",
}


def analysis_artifact_paths(recording_id: str) -> dict[AnalysisKind, str]:
    prefix = recording_id.strip().strip("/")
    return {
        kind: f"{prefix}/analysis/{filename}"
        for kind, filename in ANALYSIS_FILENAMES.items()
    }


def gaussian_splat_dir(recording_id: str) -> str:
    """Return the storage directory prefix for the gaussian splat artifacts."""
    prefix = recording_id.strip().strip("/")
    return f"{prefix}/analysis/gaussian_splat"


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
