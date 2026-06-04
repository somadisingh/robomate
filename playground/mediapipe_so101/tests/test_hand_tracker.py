from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mediapipe_so101.hand_tracker import LatestHandResult, draw_sample, ensure_model
from mediapipe_so101.types import HandSample, Landmark


def point(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def handedness(label: str, score: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(category_name=label, score=score)]


def result(
    hand_landmarks: list[list[SimpleNamespace]],
    handedness_entries: list[list[SimpleNamespace]],
) -> SimpleNamespace:
    return SimpleNamespace(hand_landmarks=hand_landmarks, handedness=handedness_entries)


def latest_sample(fake_result: SimpleNamespace) -> HandSample | None:
    latest = LatestHandResult()
    latest.update(fake_result, SimpleNamespace(), 1234)
    return latest.best_sample()


def test_best_sample_defaults_label_and_confidence_when_handedness_missing() -> None:
    sample = latest_sample(result([[point() for _ in range(21)]], []))

    assert sample is not None
    assert sample.handedness == "Hand"
    assert sample.confidence == 0.0
    assert sample.timestamp_ms == 1234


def test_best_sample_allows_fewer_handedness_entries_than_landmark_lists() -> None:
    low_score_hand = [point(0.1) for _ in range(21)]
    missing_handedness_hand = [point(0.9) for _ in range(21)]

    sample = latest_sample(result([low_score_hand, missing_handedness_hand], [handedness("Left", 0.3)]))

    assert sample is not None
    assert sample.handedness == "Left"
    assert sample.confidence == 0.3


def test_best_sample_ignores_extra_handedness_entries() -> None:
    only_hand = [point(0.2) for _ in range(21)]

    sample = latest_sample(
        result(
            [only_hand],
            [
                handedness("Left", 0.2),
                handedness("Right", 0.99),
            ],
        )
    )

    assert sample is not None
    assert sample.handedness == "Left"
    assert sample.confidence == 0.2
    assert sample.landmarks[0].x == 0.2


def test_draw_sample_allows_short_partial_landmarks() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    sample = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0)],
        handedness="Hand",
        confidence=0.0,
        timestamp_ms=1,
    )

    draw_sample(frame, sample)


def test_ensure_model_removes_tmp_file_when_download_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "models" / "hand_landmarker.task"
    tmp_model_path = model_path.with_suffix(model_path.suffix + ".tmp")
    tmp_model_path.parent.mkdir(parents=True)
    tmp_model_path.write_bytes(b"stale")

    def fail_download(*_args, **_kwargs):
        raise TimeoutError("download timed out")

    monkeypatch.setattr("mediapipe_so101.hand_tracker.urllib.request.urlopen", fail_download)

    with pytest.raises(TimeoutError, match="download timed out"):
        ensure_model(model_path)

    assert not model_path.exists()
    assert not tmp_model_path.exists()
