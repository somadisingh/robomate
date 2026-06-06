"""Unit tests for the Gemini temporal annotator + Pinecone integration.

All external services (Gemini Files/Models, Pinecone, Supabase) are mocked via
dependency injection so these run fully offline.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from backend.analyzers import gemini_temporal_annotator as gta
from backend.contracts import AnalysisRequest, TaskRecord
from backend.orchestrator import run_temporal_annotation_step


VALID_ANNOTATIONS: dict[str, Any] = {
    "duration_seconds": 12.5,
    "task_completed": True,
    "completion_confidence": 0.9,
    "scene_summary": "A right hand picks up a bottle and places it on the table.",
    "key_moments": [
        {
            "timestamp_seconds": 1.0,
            "action": "reaches for bottle",
            "objects_involved": ["bottle"],
            "hand": "right",
            "phase": "approach",
        },
        {
            "timestamp_seconds": 3.5,
            "action": "grasps bottle",
            "objects_involved": ["bottle"],
            "hand": "right",
            "phase": "grasp",
        },
    ],
    "object_visibility": {
        "bottle": {
            "first_visible_at": 0.5,
            "last_visible_at": 12.0,
            "total_visible_seconds": 11.5,
        }
    },
    "failure_moments": [],
    "searchable_tags": ["bottle", "pick-and-place", "table"],
}


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeFile:
    def __init__(self, name: str = "files/test") -> None:
        self.name = name
        self.state = None  # _state_name("") -> treated as ACTIVE immediately


class _FakeFiles:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def upload(self, *, file: str) -> _FakeFile:  # noqa: A002 - mirror SDK kwarg
        return _FakeFile()

    def get(self, *, name: str) -> _FakeFile:
        return _FakeFile(name)

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, response_text: str = "", embedding: list[float] | None = None):
        self._response_text = response_text
        self._embedding = embedding or []
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any = None):
        self.generate_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return _FakeResponse(self._response_text)

    def embed_content(self, *, model: str, contents: Any, config: Any = None):
        self.embed_calls.append({"model": model, "contents": contents})

        class _Emb:
            def __init__(self, values: list[float]) -> None:
                self.values = values

        class _Result:
            def __init__(self, values: list[float]) -> None:
                self.embeddings = [_Emb(values)]

        return _Result(self._embedding)


class _FakeGeminiClient:
    def __init__(self, response_text: str = "", embedding: list[float] | None = None):
        self.files = _FakeFiles()
        self.models = _FakeModels(response_text=response_text, embedding=embedding)


class _FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[Any] = []

    def upsert(self, *, vectors: list[dict[str, Any]]) -> None:
        self.upserts.append(vectors)


class _FakeIndexInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePinecone:
    def __init__(self, existing: list[str] | None = None) -> None:
        self._existing = existing or []
        self.index = _FakeIndex()
        self.created: list[dict[str, Any]] = []

    def list_indexes(self) -> list[_FakeIndexInfo]:
        return [_FakeIndexInfo(n) for n in self._existing]

    def create_index(self, **kwargs: Any) -> None:
        self.created.append(kwargs)
        self._existing.append(kwargs["name"])

    def Index(self, name: str) -> _FakeIndex:  # noqa: N802 - mirror SDK name
        return self.index


class _FakeApi:
    """Minimal stand-in for SupabaseApi capturing job/storage writes."""

    def __init__(self) -> None:
        self.patches: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

    def patch_rows(self, table: str, query: str, payload: dict[str, Any]):
        self.patches.append({"table": table, "query": query, "payload": payload})
        return []

    def upsert_rows(self, table: str, payload: list[dict[str, Any]], *, on_conflict: str):
        self.upserts.append({"table": table, "payload": payload})
        return payload

    def upload_json(self, bucket: str, path: str, payload: Any) -> None:
        self.uploads.append({"bucket": bucket, "path": path, "payload": payload})


# --------------------------------------------------------------------------- #
# Test 1 — Annotation prompt correctness
# --------------------------------------------------------------------------- #


def test_run_temporal_annotation_returns_full_schema(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    client = _FakeGeminiClient(response_text=json.dumps(VALID_ANNOTATIONS))

    result = gta.run_temporal_annotation(
        recording_id="rec-1",
        video_path=str(video),
        task_title="Pick up bottle",
        task_description="Pick up the bottle and place it on the table.",
        objects_to_detect=["bottle", "table"],
        client=client,
        poll_interval_s=0,
    )

    for key in (
        "duration_seconds",
        "task_completed",
        "key_moments",
        "object_visibility",
        "failure_moments",
        "searchable_tags",
        "scene_summary",
    ):
        assert key in result

    assert isinstance(result["key_moments"], list)
    for moment in result["key_moments"]:
        for field in ("timestamp_seconds", "action", "phase", "hand", "objects_involved"):
            assert field in moment

    # The uploaded file is cleaned up afterwards.
    assert client.files.deleted, "expected the uploaded Gemini file to be deleted"
    # The prompt embeds the task metadata.
    prompt = client.models.generate_calls[0]["contents"][1]
    assert "Pick up bottle" in prompt
    assert "bottle" in prompt


# --------------------------------------------------------------------------- #
# Test 2 — JSON parsing resilience (markdown fences)
# --------------------------------------------------------------------------- #


def test_run_temporal_annotation_strips_markdown_fences(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    fenced = "```json\n" + json.dumps(VALID_ANNOTATIONS) + "\n```"
    client = _FakeGeminiClient(response_text=fenced)

    result = gta.run_temporal_annotation(
        recording_id="rec-2",
        video_path=str(video),
        task_title="Pick up bottle",
        task_description="desc",
        objects_to_detect=["bottle"],
        client=client,
        poll_interval_s=0,
    )
    assert result["task_completed"] is True
    assert len(result["key_moments"]) == 2


# --------------------------------------------------------------------------- #
# Test 3 — Embedding document construction
# --------------------------------------------------------------------------- #


def test_build_pinecone_embedding(tmp_path):
    client = _FakeGeminiClient(embedding=[0.1] * gta.EMBEDDING_DIMENSION)

    payload = gta.build_pinecone_embedding(
        recording_id="rec-3",
        task_title="Pick up bottle",
        task_description="Pick up the bottle and place it on the table.",
        objects_to_detect=["bottle", "table"],
        annotations=VALID_ANNOTATIONS,
        gemini_score=8.0,
        client=client,
    )

    assert payload["id"] == "rec-3"
    assert len(payload["values"]) == gta.EMBEDDING_DIMENSION
    assert "metadata" in payload
    assert payload["metadata"]["task_title"] == "Pick up bottle"
    doc = payload["metadata"]["embedding_document"]
    assert isinstance(doc, str) and 0 < len(doc) <= 1000


# --------------------------------------------------------------------------- #
# Test 4 — Pinecone upsert
# --------------------------------------------------------------------------- #


def test_upsert_to_pinecone_calls_index_once():
    pc = _FakePinecone(existing=["robomate"])
    payload = {"id": "rec-4", "values": [0.0] * 768, "metadata": {}}

    gta.upsert_to_pinecone(payload, client=pc, index_name="robomate")

    assert len(pc.index.upserts) == 1
    vectors = pc.index.upserts[0]
    assert vectors[0]["id"] == "rec-4"
    # No index creation when it already exists.
    assert pc.created == []


# --------------------------------------------------------------------------- #
# Test 5 — Pipeline failure isolation
# --------------------------------------------------------------------------- #


def test_temporal_step_failure_is_isolated(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(gta, "run_temporal_annotation", _boom)

    api = _FakeApi()
    request = AnalysisRequest(
        recording_id="rec-5", task_id="task-5", storage_path="rec-5"
    )
    task = TaskRecord(id="task-5", title="Pick up bottle", objects=["bottle"])

    # Must NOT raise even though the annotation step blows up.
    result = run_temporal_annotation_step(api, request, task, b"video-bytes", 7.0)

    assert result is None
    statuses = [
        p["payload"].get("status")
        for p in api.patches
        if "gemini_temporal_annotations" in p["query"]
    ]
    assert "failed" in statuses
    assert "succeeded" not in statuses
    # Nothing was uploaded since the annotation failed.
    assert api.uploads == []


# --------------------------------------------------------------------------- #
# Test 6 — Pinecone index creation on first run
# --------------------------------------------------------------------------- #


def test_upsert_to_pinecone_creates_index_when_missing():
    pc = _FakePinecone(existing=[])
    payload = {"id": "rec-6", "values": [0.0] * 768, "metadata": {}}

    gta.upsert_to_pinecone(payload, client=pc, index_name="robomate")

    assert len(pc.created) == 1
    created = pc.created[0]
    assert created["dimension"] == 768
    assert created["metric"] == "cosine"
    assert len(pc.index.upserts) == 1
