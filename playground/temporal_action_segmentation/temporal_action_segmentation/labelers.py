from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporal_action_segmentation.segmentation import Segment


PROMPT_VERSION = "tas-v1"


@dataclass(frozen=True)
class LabelResult:
    meaningful_manipulation: bool
    caption: str
    object: str | None
    confidence: float
    reason: str
    raw_response: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "meaningful_manipulation": self.meaningful_manipulation,
            "caption": self.caption,
            "object": self.object,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class NullLabeler:
    def label(self, segment: Segment, contact_sheet_path: Path) -> LabelResult:
        return LabelResult(
            meaningful_manipulation=False,
            caption="N/A",
            object=None,
            confidence=0.0,
            reason="No VLM labeler selected.",
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in response: {text[:200]}")
    return json.loads(stripped[start : end + 1])


def _normalize_label(payload: dict[str, Any], raw_response: str | None = None) -> LabelResult:
    meaningful = bool(payload.get("meaningful_manipulation", False))
    caption = str(payload.get("caption") or "N/A").strip()
    if not meaningful:
        caption = "N/A"
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    confidence = min(1.0, max(0.0, confidence))
    obj = payload.get("object")
    return LabelResult(
        meaningful_manipulation=meaningful,
        caption=caption,
        object=None if obj in ("", "null", None) else str(obj),
        confidence=confidence,
        reason=str(payload.get("reason") or ""),
        raw_response=raw_response,
    )


class OpenAILabeler:
    def __init__(self, model: str, cache_dir: Path) -> None:
        self.model = model
        self.cache_dir = cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def label(self, segment: Segment, contact_sheet_path: Path) -> LabelResult:
        cache_path = self._cache_path(contact_sheet_path, segment)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return _normalize_label(cached["label"], cached.get("raw_response"))

        from openai import OpenAI

        client = OpenAI()
        image_bytes = contact_sheet_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = self._prompt(segment)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded}",
                        },
                    ],
                }
            ],
        )
        raw_text = response.output_text
        payload = _extract_json(raw_text)
        label = _normalize_label(payload, raw_text)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "contact_sheet": str(contact_sheet_path),
                    "label": label.as_record(),
                    "raw_response": raw_text,
                },
                indent=2,
            )
        )
        return label

    def _cache_path(self, contact_sheet_path: Path, segment: Segment) -> Path:
        digest = hashlib.sha256()
        digest.update(PROMPT_VERSION.encode("utf-8"))
        digest.update(self.model.encode("utf-8"))
        digest.update(segment.hand.encode("utf-8"))
        digest.update(str(segment.start_frame).encode("ascii"))
        digest.update(str(segment.end_frame).encode("ascii"))
        digest.update(contact_sheet_path.read_bytes())
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _prompt(self, segment: Segment) -> str:
        return f"""You are annotating egocentric hand-manipulation video clips.
The colored trajectory marks the {segment.hand} hand over this clip.
Describe only the action performed by that hand.
Use an imperative robot-instruction style, for example "Pick up the mug", "Open the drawer", "Wipe the counter".

Return strict JSON only:
{{
  "meaningful_manipulation": true,
  "caption": "short imperative caption or N/A",
  "object": "object name or null",
  "confidence": 0.0,
  "reason": "short reason"
}}

Rules:
- If the hand is idle, gesturing, occluded, or not manipulating an object, return meaningful_manipulation false and caption "N/A".
- Prefer short atomic actions.
- Do not describe camera motion.
- Do not invent objects that are not visible.
- The clip spans {segment.start_sec:.2f}s to {segment.end_sec:.2f}s in the source video."""


def make_labeler(kind: str, model: str, cache_dir: Path):
    if kind == "none":
        return NullLabeler()
    if kind == "openai":
        return OpenAILabeler(model=model, cache_dir=cache_dir)
    raise ValueError(f"Unsupported labeler: {kind}")
