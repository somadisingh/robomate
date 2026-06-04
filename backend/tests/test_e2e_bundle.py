import pytest

from backend.tools.e2e_upload_bundle import (
    build_submit_payload,
    ensure_analysis_started,
    normalize_recording_id,
    patch_metadata,
    poll_all_jobs,
    poll_score,
    resource_intensive_analysis_enabled,
)


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


def test_ensure_analysis_started_fails_on_explicit_false():
    with pytest.raises(RuntimeError, match="Modal analysis env is not configured"):
        ensure_analysis_started(
            {
                "analysis_started": False,
                "analysis_error": "Modal analysis env is not configured",
            }
        )


def test_normalize_recording_id_requires_uuid():
    assert (
        normalize_recording_id("E1E838FC-4506-4FEE-9E75-32C35F7B9460")
        == "e1e838fc-4506-4fee-9e75-32c35f7b9460"
    )

    with pytest.raises(ValueError, match="recording_id must be a UUID"):
        normalize_recording_id("e2e-not-a-uuid")


def test_poll_score_fails_when_scoring_done_without_fields():
    api = FakeApi(
        recording={
            "id": "rec",
            "is_scoring": False,
            "summary": "",
            "success": None,
            "score": None,
        },
        jobs=[
            {
                "kind": "gemini_eval",
                "status": "failed",
                "artifact_path": "rec/analysis/gemini-eval.json",
                "error": "Gemini quota exceeded",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Gemini quota exceeded"):
        poll_score(api, "rec", timeout_s=10)


def test_poll_all_jobs_fails_on_terminal_failed_jobs():
    api = FakeApi(
        jobs=[
            {
                "kind": "gemini_eval",
                "status": "succeeded",
                "artifact_path": "rec/analysis/gemini-eval.json",
                "error": None,
            },
            {
                "kind": "mediapipe_hands",
                "status": "failed",
                "artifact_path": "rec/analysis/mediapipe-hands.json",
                "error": "MediaPipe crashed",
            },
            {
                "kind": "yolo_objects",
                "status": "succeeded",
                "artifact_path": "rec/analysis/yolo-detections.json",
                "error": None,
            },
            {
                "kind": "sam_segments",
                "status": "succeeded",
                "artifact_path": "rec/analysis/sam-segments.json",
                "error": None,
            },
            {
                "kind": "temporal_actions",
                "status": "succeeded",
                "artifact_path": "rec/analysis/temporal-actions.json",
                "error": None,
            },
        ],
    )

    with pytest.raises(RuntimeError, match="mediapipe_hands: MediaPipe crashed"):
        poll_all_jobs(api, "rec", timeout_s=10)

    assert api.downloaded == []


def test_resource_intensive_analysis_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS", raising=False)

    assert resource_intensive_analysis_enabled() is False


def test_resource_intensive_analysis_flag_accepts_truthy(monkeypatch):
    monkeypatch.setenv("COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS", "1")

    assert resource_intensive_analysis_enabled() is True


class FakeResponse:
    pass


class FakeClient:
    def __init__(self, api):
        self.api = api

    def get(self, *_args, **_kwargs):
        return FakeResponse()


class FakeConfig:
    url = "https://example.supabase.co"


class FakeApi:
    config = FakeConfig()

    def __init__(self, *, recording=None, jobs=None):
        self.recording = recording or {}
        self.jobs = jobs or []
        self.client = FakeClient(self)
        self.downloaded = []

    def select_one(self, table, query):
        assert table == "recordings"
        assert "id=eq.rec" in query
        return self.recording

    def rest_headers(self):
        return {}

    def _json(self, _res):
        return self.jobs

    def download_bytes(self, bucket, path):
        self.downloaded.append((bucket, path))
        return b"{}"
