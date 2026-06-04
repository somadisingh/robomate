from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np

from temporal_action_segmentation.models import HandTrack, VideoSample, VideoTracks


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PALM_LANDMARKS = (0, 5, 9, 13, 17)
HAND_NAMES = ("left", "right")
_LIVE_LANDMARKERS = []


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def ensure_model(model_path: Path | None = None) -> Path:
    path = (model_path or default_model_path()).expanduser().resolve()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading MediaPipe Hand Landmarker model to {path}")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(path)
    return path


def create_landmarker(
    model_path: Path,
    max_hands: int = 2,
    detection_confidence: float = 0.5,
    presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def _open_video(video_path: Path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def _palm_center(hand_landmarks) -> tuple[float, float]:
    xy = np.array(
        [(hand_landmarks[index].x, hand_landmarks[index].y) for index in PALM_LANDMARKS],
        dtype=float,
    )
    return float(xy[:, 0].mean()), float(xy[:, 1].mean())


def _best_hands_by_label(result) -> dict[str, tuple[tuple[float, float], float]]:
    best: dict[str, tuple[tuple[float, float], float]] = {}
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[hand_index] if hand_index < len(result.handedness) else []
        if not handedness:
            label = "hand"
            score = 0.0
        else:
            category = handedness[0]
            label = str(category.category_name).lower()
            score = float(category.score)
        if label not in HAND_NAMES:
            continue
        if label not in best or score > best[label][1]:
            best[label] = (_palm_center(landmarks), score)
    return best


def track_video(
    video_path: Path,
    model_path: Path,
    target_fps: float = 10.0,
    max_hands: int = 2,
    detection_confidence: float = 0.5,
    presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
) -> VideoTracks:
    import cv2
    import mediapipe as mp

    video_path = video_path.expanduser().resolve()
    capture = _open_video(video_path)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / target_fps)))
    sample_fps = source_fps / stride

    samples: list[VideoSample] = []
    points = {hand: [] for hand in HAND_NAMES}
    confidences = {hand: [] for hand in HAND_NAMES}

    try:
        landmarker = create_landmarker(
            model_path=model_path,
            max_hands=max_hands,
            detection_confidence=detection_confidence,
            presence_confidence=presence_confidence,
            tracking_confidence=tracking_confidence,
        )
        # On macOS with MediaPipe 0.10.35, close() can hang after detect_for_video().
        # Keep the task alive until process exit rather than triggering close/destruct.
        _LIVE_LANDMARKERS.append(landmarker)
        source_frame = 0
        sample_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame % stride != 0:
                source_frame += 1
                continue

            time_sec = source_frame / source_fps
            timestamp_ms = int(round(time_sec * 1000.0))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands = _best_hands_by_label(result)

            samples.append(VideoSample(sample_index, source_frame, time_sec))
            for hand in HAND_NAMES:
                if hand in hands:
                    center, score = hands[hand]
                    points[hand].append(center)
                    confidences[hand].append(score)
                else:
                    points[hand].append((np.nan, np.nan))
                    confidences[hand].append(np.nan)

            sample_index += 1
            source_frame += 1
    finally:
        capture.release()

    tracks = {
        hand: HandTrack(
            hand=hand,
            fps=sample_fps,
            points=np.asarray(points[hand], dtype=float),
            confidence=np.asarray(confidences[hand], dtype=float),
        )
        for hand in HAND_NAMES
    }
    print(
        f"Tracked {video_path.name}: {len(samples)} samples at {sample_fps:.2f} fps "
        f"from {frame_count or 'unknown'} source frames"
    )
    return VideoTracks(
        video_id=video_path.stem,
        video_path=video_path,
        width=width,
        height=height,
        source_fps=source_fps,
        sample_fps=sample_fps,
        samples=samples,
        tracks=tracks,
    )


def write_tracks_csv(video_tracks: VideoTracks, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "source_frame",
                "time_sec",
                "left_x",
                "left_y",
                "left_confidence",
                "left_visible",
                "right_x",
                "right_y",
                "right_confidence",
                "right_visible",
            ],
        )
        writer.writeheader()
        for sample in video_tracks.samples:
            row = {
                "sample_index": sample.index,
                "source_frame": sample.source_frame,
                "time_sec": f"{sample.time_sec:.6f}",
            }
            for hand in HAND_NAMES:
                track = video_tracks.tracks[hand]
                point = track.points[sample.index]
                row[f"{hand}_x"] = "" if not np.isfinite(point[0]) else f"{point[0]:.6f}"
                row[f"{hand}_y"] = "" if not np.isfinite(point[1]) else f"{point[1]:.6f}"
                confidence = track.confidence[sample.index]
                row[f"{hand}_confidence"] = "" if not np.isfinite(confidence) else f"{confidence:.6f}"
                row[f"{hand}_visible"] = bool(track.visible[sample.index])
            writer.writerow(row)
    return output_path
