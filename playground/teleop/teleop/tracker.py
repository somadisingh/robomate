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

from .types import ArmSample, HandSample, Landmark, PoseLandmark, TeleopSample


POSE_RIGHT_SHOULDER = 12
POSE_RIGHT_ELBOW = 14
POSE_RIGHT_WRIST = 16
POSE_LEFT_SHOULDER = 11
POSE_LEFT_ELBOW = 13
POSE_LEFT_WRIST = 15


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def default_pose_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"


def default_hand_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def ensure_model(model_path: Path, *, url: str) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"Downloading MediaPipe model to {model_path}")
    tmp_path.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with tmp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp_path.replace(model_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return model_path


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
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        capture.read()
    return capture


def frame_to_mp_image(frame) -> mp.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def create_pose_landmarker(
    *,
    model_path: Path,
    detection_confidence: float,
    presence_confidence: float,
    tracking_confidence: float,
    result_callback,
) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        output_segmentation_masks=False,
        result_callback=result_callback,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_landmarker(
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


class LatestPoseResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None
        self._timestamp_ms = -1

    def update(self, result, _output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def best_arm_sample(self, arm: str) -> ArmSample | None:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        if result is None or not result.pose_landmarks or not result.pose_world_landmarks:
            return None

        if arm == "right":
            indices = (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST)
        elif arm == "left":
            indices = (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST)
        else:
            raise ValueError(f"Unknown arm selection: {arm}")

        image_lms = result.pose_landmarks[0]
        world_lms = result.pose_world_landmarks[0]
        shoulder_world = world_lms[indices[0]]
        elbow_world = world_lms[indices[1]]
        wrist_world = world_lms[indices[2]]
        shoulder_image = image_lms[indices[0]]
        elbow_image = image_lms[indices[1]]
        wrist_image = image_lms[indices[2]]

        return ArmSample(
            shoulder=PoseLandmark(
                shoulder_world.x,
                shoulder_world.y,
                shoulder_world.z,
                visibility=shoulder_world.visibility,
            ),
            shoulder_image_xy=(shoulder_image.x, shoulder_image.y),
            elbow=PoseLandmark(
                elbow_world.x, elbow_world.y, elbow_world.z, visibility=elbow_world.visibility
            ),
            elbow_image_xy=(elbow_image.x, elbow_image.y),
            wrist=PoseLandmark(
                wrist_world.x, wrist_world.y, wrist_world.z, visibility=wrist_world.visibility
            ),
            wrist_image_xy=(wrist_image.x, wrist_image.y),
            timestamp_ms=timestamp_ms,
        )


class LatestHandResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = None
        self._timestamp_ms = -1

    def update(self, result, _output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def best_hand_sample(self) -> HandSample | None:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        return _extract_best_hand(result, timestamp_ms)

    def all_hand_samples(self) -> list[HandSample]:
        with self._lock:
            result = self._result
            timestamp_ms = self._timestamp_ms
        if result is None or not result.hand_landmarks:
            return []
        samples: list[HandSample] = []
        for idx in range(len(result.hand_landmarks)):
            handedness = result.handedness[idx] if idx < len(result.handedness) else []
            label = handedness[0].category_name if handedness else "Hand"
            score = handedness[0].score if handedness else 0.0
            landmarks = [
                Landmark(x=point.x, y=point.y, z=point.z)
                for point in result.hand_landmarks[idx]
            ]
            samples.append(
                HandSample(
                    landmarks=landmarks,
                    handedness=label,
                    confidence=score,
                    timestamp_ms=timestamp_ms,
                )
            )
        return samples


def _extract_best_hand(result, timestamp_ms: int) -> HandSample | None:
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


def fuse_samples(
    *, arm: ArmSample | None, hands: list[HandSample], timestamp_ms: int
) -> TeleopSample:
    if arm is None:
        chosen_hand = max(hands, key=lambda h: h.confidence, default=None)
        fused_ts = chosen_hand.timestamp_ms if chosen_hand is not None else timestamp_ms
        return TeleopSample(arm=None, hand=chosen_hand, timestamp_ms=fused_ts)

    if not hands:
        return TeleopSample(arm=arm, hand=None, timestamp_ms=arm.timestamp_ms)

    wrist_x, wrist_y = arm.wrist_image_xy

    def distance(hand: HandSample) -> float:
        wrist_landmark = hand.landmarks[0]
        return (wrist_landmark.x - wrist_x) ** 2 + (wrist_landmark.y - wrist_y) ** 2

    chosen_hand = min(hands, key=distance)
    # Use the older of the two underlying sample timestamps so the downstream
    # stale-result check is conservative: if either modality is lagging, we
    # report the lag instead of masking it with the loop clock.
    fused_ts = min(arm.timestamp_ms, chosen_hand.timestamp_ms)
    return TeleopSample(arm=arm, hand=chosen_hand, timestamp_ms=fused_ts)


HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


def draw_overlay(
    frame,
    *,
    arm: ArmSample | None,
    hand: HandSample | None,
    status_lines: list[str],
    image_size: tuple[int, int],
    arm_image_landmarks: dict[str, tuple[float, float]] | None = None,
) -> None:
    width, height = image_size

    if arm is not None and arm_image_landmarks is not None:
        labels = ("shoulder", "elbow", "wrist")
        visibilities = (arm.shoulder.visibility, arm.elbow.visibility, arm.wrist.visibility)
        points = []
        for label in labels:
            normalized = arm_image_landmarks.get(label)
            if normalized is None:
                points.append(None)
                continue
            px = max(0, min(width - 1, int(normalized[0] * width)))
            py = max(0, min(height - 1, int(normalized[1] * height)))
            points.append((px, py))
        for start, end in ((0, 1), (1, 2)):
            if points[start] is None or points[end] is None:
                continue
            cv2.line(frame, points[start], points[end], (255, 180, 70), 3, cv2.LINE_AA)
        for label, point, visibility in zip(labels, points, visibilities):
            if point is None:
                continue
            cv2.circle(frame, point, 7, (255, 220, 130), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{visibility:.2f}",
                (point[0] + 8, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 130),
                1,
                cv2.LINE_AA,
            )

    if hand is not None:
        hand_points = [
            (
                max(0, min(width - 1, int(point.x * width))),
                max(0, min(height - 1, int(point.y * height))),
            )
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            if start >= len(hand_points) or end >= len(hand_points):
                continue
            cv2.line(frame, hand_points[start], hand_points[end], (65, 210, 120), 2, cv2.LINE_AA)
        for index, point in enumerate(hand_points):
            radius = 6 if index in {4, 8, 12, 16, 20} else 4
            cv2.circle(frame, point, radius, (35, 115, 255), -1, cv2.LINE_AA)

    for index, line in enumerate(status_lines):
        y = 30 + index * 22
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
