import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from teleop.tracker import (
    LatestHandResult,
    LatestPoseResult,
    default_hand_model_path,
    default_pose_model_path,
    draw_overlay,
    ensure_model,
    fuse_samples,
)
from teleop.types import ArmSample, HandSample, Landmark, PoseLandmark


def test_default_pose_model_path_lives_under_models_dir() -> None:
    path = default_pose_model_path()
    assert path.name == "pose_landmarker_lite.task"
    assert path.parent.name == "models"


def test_default_hand_model_path_lives_under_models_dir() -> None:
    path = default_hand_model_path()
    assert path.name == "hand_landmarker.task"
    assert path.parent.name == "models"


def test_ensure_model_returns_existing_path_when_file_present(tmp_path: Path) -> None:
    file_path = tmp_path / "existing.task"
    file_path.write_bytes(b"\x00")
    assert ensure_model(file_path, url="https://example.invalid/model.task") == file_path.resolve()


def test_ensure_model_downloads_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "missing.task"

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self._sent = False

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args, **kwargs) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return self._payload

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        assert url == "https://example.invalid/model.task"
        return FakeResponse(b"abc")

    with patch("teleop.tracker.urllib.request.urlopen", side_effect=fake_urlopen):
        result = ensure_model(target, url="https://example.invalid/model.task")
    assert result == target.resolve()
    assert target.read_bytes() == b"abc"


def test_ensure_model_cleans_up_temp_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "fail.task"

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    with patch("teleop.tracker.urllib.request.urlopen", side_effect=boom):
        with pytest.raises(RuntimeError, match="network down"):
            ensure_model(target, url="https://example.invalid/model.task")
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def make_pose_landmark_obj(x: float, y: float, z: float, visibility: float):
    class _L:
        pass

    obj = _L()
    obj.x = x
    obj.y = y
    obj.z = z
    obj.visibility = visibility
    obj.presence = visibility
    return obj


def make_pose_result(
    *,
    image_landmarks_per_arm: dict[str, tuple[float, float, float, float]] | None = None,
    world_landmarks_per_arm: dict[str, tuple[float, float, float, float]] | None = None,
) -> object:
    class _Result:
        pass

    result = _Result()
    image_defaults = {
        "shoulder": (0.5, 0.3, 0.0, 0.95),
        "elbow": (0.6, 0.5, 0.0, 0.95),
        "wrist": (0.7, 0.7, 0.0, 0.95),
    }
    world_defaults = {
        "shoulder": (0.0, 0.0, 0.0, 0.95),
        "elbow": (0.1, 0.2, 0.0, 0.95),
        "wrist": (0.15, 0.4, 0.0, 0.95),
    }
    image_data = image_landmarks_per_arm or image_defaults
    world_data = world_landmarks_per_arm or world_defaults

    image_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    world_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    image_landmarks[12] = make_pose_landmark_obj(*image_data["shoulder"])
    image_landmarks[14] = make_pose_landmark_obj(*image_data["elbow"])
    image_landmarks[16] = make_pose_landmark_obj(*image_data["wrist"])
    world_landmarks[12] = make_pose_landmark_obj(*world_data["shoulder"])
    world_landmarks[14] = make_pose_landmark_obj(*world_data["elbow"])
    world_landmarks[16] = make_pose_landmark_obj(*world_data["wrist"])

    result.pose_landmarks = [image_landmarks]
    result.pose_world_landmarks = [world_landmarks]
    return result


def make_hand_result(hands: list[dict]) -> object:
    class _Result:
        pass

    result = _Result()
    hand_landmarks_lists = []
    handedness_lists = []
    for hand in hands:
        points = [make_pose_landmark_obj(hand["x"], hand["y"], 0.0, 0.0) for _ in range(21)]
        hand_landmarks_lists.append(points)
        handedness_objs = []

        class _H:
            pass

        h = _H()
        h.category_name = hand.get("handedness", "Right")
        h.score = hand.get("score", 0.9)
        handedness_objs.append(h)
        handedness_lists.append(handedness_objs)
    result.hand_landmarks = hand_landmarks_lists
    result.handedness = handedness_lists
    return result


def test_latest_pose_result_returns_none_when_no_callback_received() -> None:
    latest = LatestPoseResult()
    assert latest.best_arm_sample(arm="right") is None


def test_latest_pose_result_returns_arm_sample_for_chosen_arm() -> None:
    latest = LatestPoseResult()
    latest.update(make_pose_result(), None, timestamp_ms=10)

    sample = latest.best_arm_sample(arm="right")
    assert sample is not None
    assert sample.timestamp_ms == 10
    assert sample.shoulder.visibility == pytest.approx(0.95)
    assert sample.wrist_image_xy == pytest.approx((0.7, 0.7))


def test_best_arm_sample_populates_all_image_xy_fields() -> None:
    latest = LatestPoseResult()
    latest.update(make_pose_result(), None, timestamp_ms=10)

    sample = latest.best_arm_sample(arm="right")
    assert sample is not None
    assert sample.shoulder_image_xy == pytest.approx((0.5, 0.3))
    assert sample.elbow_image_xy == pytest.approx((0.6, 0.5))
    assert sample.wrist_image_xy == pytest.approx((0.7, 0.7))


def test_latest_pose_result_supports_left_arm() -> None:
    latest = LatestPoseResult()
    image_data = {
        "shoulder": (0.4, 0.3, 0.0, 0.8),
        "elbow": (0.3, 0.5, 0.0, 0.8),
        "wrist": (0.2, 0.7, 0.0, 0.8),
    }
    world_data = {
        "shoulder": (0.0, 0.0, 0.0, 0.8),
        "elbow": (-0.1, 0.2, 0.0, 0.8),
        "wrist": (-0.15, 0.4, 0.0, 0.8),
    }
    # Left side indices: shoulder=11, elbow=13, wrist=15
    class _Result:
        pass

    result = _Result()
    image_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    world_landmarks = [make_pose_landmark_obj(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    image_landmarks[11] = make_pose_landmark_obj(*image_data["shoulder"])
    image_landmarks[13] = make_pose_landmark_obj(*image_data["elbow"])
    image_landmarks[15] = make_pose_landmark_obj(*image_data["wrist"])
    world_landmarks[11] = make_pose_landmark_obj(*world_data["shoulder"])
    world_landmarks[13] = make_pose_landmark_obj(*world_data["elbow"])
    world_landmarks[15] = make_pose_landmark_obj(*world_data["wrist"])
    result.pose_landmarks = [image_landmarks]
    result.pose_world_landmarks = [world_landmarks]

    latest.update(result, None, timestamp_ms=20)
    sample = latest.best_arm_sample(arm="left")
    assert sample is not None
    assert sample.wrist_image_xy == pytest.approx((0.2, 0.7))


def test_latest_hand_result_returns_none_when_no_callback_received() -> None:
    latest = LatestHandResult()
    assert latest.best_hand_sample() is None


def test_latest_hand_result_returns_highest_confidence_sample() -> None:
    latest = LatestHandResult()
    result = make_hand_result(
        [
            {"x": 0.4, "y": 0.5, "score": 0.5},
            {"x": 0.7, "y": 0.7, "score": 0.95},
        ]
    )
    latest.update(result, None, timestamp_ms=30)
    sample = latest.best_hand_sample()
    assert sample is not None
    assert sample.confidence == pytest.approx(0.95)
    assert sample.landmarks[0].x == pytest.approx(0.7)


def test_latest_result_buffers_are_thread_safe() -> None:
    latest = LatestHandResult()
    result_a = make_hand_result([{"x": 0.5, "y": 0.5, "score": 0.7}])
    result_b = make_hand_result([{"x": 0.5, "y": 0.5, "score": 0.7}])

    def writer(result, timestamp_ms):
        for _ in range(100):
            latest.update(result, None, timestamp_ms=timestamp_ms)

    threads = [
        threading.Thread(target=writer, args=(result_a, 10)),
        threading.Thread(target=writer, args=(result_b, 20)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sample = latest.best_hand_sample()
    assert sample is not None
    assert sample.timestamp_ms in (10, 20)


def test_fuse_samples_pairs_arm_with_nearest_hand_in_image_plane() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.75, 0.75),
        timestamp_ms=10,
    )
    far_hand = HandSample(
        landmarks=[Landmark(0.10, 0.10, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    near_hand = HandSample(
        landmarks=[Landmark(0.74, 0.76, 0.0) for _ in range(21)],
        handedness="Left",
        confidence=0.9,
        timestamp_ms=10,
    )

    sample = fuse_samples(arm=arm, hands=[far_hand, near_hand], timestamp_ms=10)
    assert sample.arm is arm
    assert sample.hand is near_hand


def test_fuse_samples_returns_none_hand_when_no_hands_present() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=10,
    )
    sample = fuse_samples(arm=arm, hands=[], timestamp_ms=10)
    assert sample.arm is arm
    assert sample.hand is None


def test_fuse_samples_returns_none_arm_when_arm_missing() -> None:
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    sample = fuse_samples(arm=None, hands=[hand], timestamp_ms=10)
    assert sample.arm is None
    assert sample.hand is hand


def test_fuse_samples_uses_oldest_underlying_sample_timestamp() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=500,
    )
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=700,
    )

    sample = fuse_samples(arm=arm, hands=[hand], timestamp_ms=1000)

    # Older of arm/hand timestamps is chosen so stale-result checks are conservative.
    assert sample.timestamp_ms == 500


def test_fuse_samples_uses_arm_timestamp_when_hand_missing() -> None:
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=300,
    )
    sample = fuse_samples(arm=arm, hands=[], timestamp_ms=1000)
    assert sample.timestamp_ms == 300


def test_fuse_samples_uses_hand_timestamp_when_arm_missing() -> None:
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=400,
    )
    sample = fuse_samples(arm=None, hands=[hand], timestamp_ms=1000)
    assert sample.timestamp_ms == 400


def test_fuse_samples_falls_back_to_loop_clock_when_no_samples() -> None:
    sample = fuse_samples(arm=None, hands=[], timestamp_ms=1234)
    assert sample.timestamp_ms == 1234


def test_draw_overlay_does_not_modify_when_samples_are_none() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    snapshot = frame.copy()
    draw_overlay(
        frame,
        arm=None,
        hand=None,
        status_lines=[],
        image_size=(320, 240),
    )
    assert np.array_equal(frame, snapshot)


def test_draw_overlay_draws_pose_skeleton_lines_when_arm_present() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    arm = ArmSample(
        shoulder=PoseLandmark(0.0, 0.0, 0.0, 0.9),
        shoulder_image_xy=(0.2, 0.2),
        elbow=PoseLandmark(0.1, 0.2, 0.0, 0.9),
        elbow_image_xy=(0.4, 0.4),
        wrist=PoseLandmark(0.2, 0.4, 0.0, 0.9),
        wrist_image_xy=(0.5, 0.5),
        timestamp_ms=10,
    )
    image_landmarks = {
        "shoulder": (0.2, 0.2),
        "elbow": (0.4, 0.4),
        "wrist": (0.5, 0.5),
    }
    draw_overlay(
        frame,
        arm=arm,
        hand=None,
        status_lines=["status text"],
        image_size=(320, 240),
        arm_image_landmarks=image_landmarks,
    )
    assert frame.sum() > 0


def test_draw_overlay_draws_hand_landmarks_when_hand_present() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    hand = HandSample(
        landmarks=[Landmark(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
        confidence=0.9,
        timestamp_ms=10,
    )
    draw_overlay(
        frame,
        arm=None,
        hand=hand,
        status_lines=[],
        image_size=(320, 240),
    )
    assert frame.sum() > 0
