from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from video_eval.evaluator import (
    DEFAULT_MODEL,
    VideoTaskEvaluation,
    build_evaluation_prompt,
    evaluate_video,
)


SAMPLE_VIDEO = Path(__file__).resolve().parents[2] / "data" / "1_can_noodles.mp4"


class FakeFiles:
    def __init__(self) -> None:
        self.uploaded_path: str | None = None
        self.deleted_name: str | None = None
        self.get_calls = 0

    def upload(self, *, file: str) -> SimpleNamespace:
        self.uploaded_path = file
        return SimpleNamespace(name="files/sample-video", state="PROCESSING")

    def get(self, *, name: str) -> SimpleNamespace:
        self.get_calls += 1
        assert name == "files/sample-video"
        return SimpleNamespace(name=name, state="ACTIVE")

    def delete(self, *, name: str) -> None:
        self.deleted_name = name


class FakeModels:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def generate_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
        self.request = {
            "model": model,
            "contents": contents,
            "config": config,
        }
        return SimpleNamespace(
            text=json.dumps(
                {
                    "task_succeeded": True,
                    "success_reasoning": "The hand places the can in the target area.",
                    "trajectory_score": 8,
                    "score_reasoning": "The trajectory is smooth with a small correction near the end.",
                }
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.models = FakeModels()


def test_schema_matches_requested_output_contract() -> None:
    result = VideoTaskEvaluation.model_validate(
        {
            "task_succeeded": False,
            "success_reasoning": "The object never reaches the requested destination.",
            "trajectory_score": 3,
            "score_reasoning": "The trajectory starts correctly but stops before completion.",
        }
    )

    assert result.task_succeeded is False
    assert result.trajectory_score == 3

    with pytest.raises(ValidationError):
        VideoTaskEvaluation.model_validate(
            {
                "task_succeeded": True,
                "success_reasoning": "Succeeded.",
                "trajectory_score": 11,
                "score_reasoning": "Out of range.",
            }
        )


def test_prompt_includes_task_and_scoring_contract() -> None:
    prompt = build_evaluation_prompt("Put the can into the cup.")

    assert "Put the can into the cup." in prompt
    assert "task_succeeded" in prompt
    assert "trajectory_score" in prompt
    assert "0 to 10" in prompt


def test_evaluate_video_uploads_sample_video_and_returns_structured_result() -> None:
    client = FakeClient()

    result = evaluate_video(
        video_path=SAMPLE_VIDEO,
        task_description="Move the can next to the noodles.",
        client=client,
        model=DEFAULT_MODEL,
        poll_interval_s=0,
    )

    assert result.task_succeeded is True
    assert result.trajectory_score == 8
    assert client.files.uploaded_path == str(SAMPLE_VIDEO)
    assert client.files.get_calls == 1
    assert client.files.deleted_name == "files/sample-video"
    assert client.models.request is not None
    assert client.models.request["model"] == DEFAULT_MODEL
