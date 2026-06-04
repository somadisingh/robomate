from __future__ import annotations

import shutil
import sys
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .types import HandSample, Landmark


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def ensure_model(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"Downloading Hand Landmarker model to {model_path}")
    tmp_path.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            with tmp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp_path.replace(model_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return model_path


class LatestHandResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: vision.HandLandmarkerResult | None = None
        self._timestamp_ms = -1

    def update(
        self,
        result: vision.HandLandmarkerResult,
        _output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def best_sample(self) -> HandSample | None:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        if result is None or not result.hand_landmarks:
            return None

        best_index = 0
        best_score = -1.0
        for idx in range(len(result.hand_landmarks)):
            handedness = result.handedness[idx] if idx < len(result.handedness) else []
            score = handedness[0].score if handedness else 0.0
            if score > best_score:
                best_index = idx
                best_score = score

        landmarks = [
            Landmark(x=point.x, y=point.y, z=point.z)
            for point in result.hand_landmarks[best_index]
        ]
        handedness = result.handedness[best_index] if best_index < len(result.handedness) else []
        label = handedness[0].category_name if handedness else "Hand"
        score = handedness[0].score if handedness else 0.0
        return HandSample(landmarks=landmarks, handedness=label, confidence=score, timestamp_ms=timestamp_ms)


def create_landmarker(
    *,
    model_path: Path,
    max_hands: int,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    result_callback,
) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        result_callback=result_callback,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    default_backend = getattr(cv2, "CAP_ANY", 0)
    api_preference = (
        getattr(cv2, "CAP_AVFOUNDATION", default_backend)
        if sys.platform == "darwin"
        else default_backend
    )
    capture = cv2.VideoCapture(camera_index, api_preference)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "On macOS, make sure the terminal app has camera permission."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def frame_to_mp_image(frame) -> mp.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def draw_sample(frame, sample: HandSample | None) -> None:
    if sample is None:
        return

    height, width = frame.shape[:2]
    points = [
        (max(0, min(width - 1, int(point.x * width))), max(0, min(height - 1, int(point.y * height))))
        for point in sample.landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        cv2.line(frame, points[start], points[end], (65, 210, 120), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 6 if index in {4, 8, 12, 16, 20} else 4
        cv2.circle(frame, point, radius, (35, 115, 255), -1, cv2.LINE_AA)
