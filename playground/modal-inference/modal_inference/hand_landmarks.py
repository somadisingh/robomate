from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_model(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(model_path)
    return model_path


def _category_payload(categories: list[Any]) -> tuple[str, float]:
    if not categories:
        return "Hand", 0.0
    category = categories[0]
    return str(category.category_name), float(category.score)


def _landmarks_payload(landmarks: list[Any]) -> list[dict[str, float]]:
    return [
        {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(getattr(landmark, "z", 0.0)),
        }
        for landmark in landmarks
    ]


def _result_payload(result: Any, frame_index: int, source_frame: int, time_sec: float) -> dict[str, Any]:
    hands = []
    world = getattr(result, "hand_world_landmarks", []) or []
    for index, landmarks in enumerate(result.hand_landmarks):
        label, score = _category_payload(result.handedness[index] if index < len(result.handedness) else [])
        hand: dict[str, Any] = {
            "handedness": label,
            "confidence": score,
            "landmarks": _landmarks_payload(landmarks),
        }
        if index < len(world):
            hand["world_landmarks"] = _landmarks_payload(world[index])
        hands.append(hand)

    return {
        "frame_index": frame_index,
        "source_frame": source_frame,
        "time_sec": round(time_sec, 6),
        "hands": hands,
    }


def create_landmarker(
    model_path: Path,
    *,
    running_mode: Any,
    max_hands: int,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def infer_hands(
    media_path: Path,
    model_path: Path,
    *,
    is_video: bool,
    target_fps: float,
    max_frames: int | None,
    max_hands: int,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
) -> dict[str, Any]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    if not is_video:
        frame = cv2.imread(str(media_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {media_path}")
        landmarker = create_landmarker(
            model_path,
            running_mode=vision.RunningMode.IMAGE,
            max_hands=max_hands,
            detection_confidence=detection_confidence,
            presence_confidence=presence_confidence,
            tracking_confidence=tracking_confidence,
        )
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
        finally:
            landmarker.close()
        return {"frames": [_result_payload(result, 0, 0, 0.0)], "frame_count": 1}

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {media_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(source_fps / target_fps))) if target_fps > 0 else 1
    sample_fps = source_fps / stride
    landmarker = create_landmarker(
        model_path,
        running_mode=vision.RunningMode.VIDEO,
        max_hands=max_hands,
        detection_confidence=detection_confidence,
        presence_confidence=presence_confidence,
        tracking_confidence=tracking_confidence,
    )

    records = []
    source_frame = 0
    sample_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame % stride != 0:
                source_frame += 1
                continue
            if max_frames is not None and sample_index >= max_frames:
                break

            time_sec = source_frame / source_fps
            timestamp_ms = int(round(time_sec * 1000.0))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            records.append(_result_payload(result, sample_index, source_frame, time_sec))
            sample_index += 1
            source_frame += 1
    finally:
        capture.release()
        landmarker.close()

    return {
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "stride": stride,
        "frame_count": len(records),
        "frames": records,
    }
