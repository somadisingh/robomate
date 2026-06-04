from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "MediaPipe Hands"
HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def playground_path() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return playground_path() / "data"


def default_output_path() -> Path:
    return playground_path() / "outputs" / "mediapipe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe hand landmark detection on a webcam stream."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to open. Default: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera capture width. Default: 1280.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera capture height. Default: 720.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect. Default: 2.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--presence-confidence",
        type=float,
        default=0.5,
        help="Minimum hand presence confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
        help="Path for the Hand Landmarker model asset.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror display/selfie view.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Download/load the model and exit without opening the camera.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="Process a video file instead of opening the webcam. Can be repeated.",
    )
    parser.add_argument(
        "--sample-videos",
        action="store_true",
        help="Process all .mp4 files from the shared playground/data directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_path(),
        help="Directory used with --sample-videos. Default: ../data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_path(),
        help="Directory for rendered video outputs. Default: ../outputs/mediapipe.",
    )
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")

    print(f"Downloading Hand Landmarker model to {model_path}")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(model_path)
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

    def get(self) -> tuple[vision.HandLandmarkerResult | None, int]:
        with self._lock:
            return self._result, self._timestamp_ms


def create_landmarker(
    args: argparse.Namespace,
    model_path: Path,
    result_callback=None,
    running_mode: vision.RunningMode = vision.RunningMode.LIVE_STREAM,
) -> vision.HandLandmarker:
    option_args = dict(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )
    if running_mode == vision.RunningMode.LIVE_STREAM:
        option_args["result_callback"] = result_callback
    options = vision.HandLandmarkerOptions(**option_args)
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


def clamp_pixel(value: float, limit: int) -> int:
    return max(0, min(limit - 1, int(value * limit)))


def draw_landmarks(
    frame,
    hand_landmarks,
    label: str,
    hand_index: int,
) -> None:
    height, width = frame.shape[:2]
    points = [
        (clamp_pixel(landmark.x, width), clamp_pixel(landmark.y, height))
        for landmark in hand_landmarks
    ]

    connection_color = (65, 210, 120)
    point_color = (35, 115, 255)
    wrist_color = (255, 190, 80)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], connection_color, 2, cv2.LINE_AA)

    for landmark_index, point in enumerate(points):
        radius = 6 if landmark_index in {4, 8, 12, 16, 20} else 4
        color = wrist_color if landmark_index == 0 else point_color
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    text_origin = (
        max(8, min(point[0] for point in points)),
        max(26, min(point[1] for point in points) - 12),
    )
    label_text = f"{hand_index + 1}: {label}"
    cv2.putText(
        frame,
        label_text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label_text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_result(frame, result: vision.HandLandmarkerResult) -> None:
    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[hand_index] if hand_index < len(result.handedness) else []
        if handedness:
            category = handedness[0]
            label = f"{category.category_name} {category.score:.2f}"
        else:
            label = "Hand"
        draw_landmarks(frame, hand_landmarks, label, hand_index)


def draw_status(frame, fps: float, hands_detected: int, result_age_ms: int | None) -> None:
    async_status = "waiting" if result_age_ms is None else f"result age: {result_age_ms}ms"
    status = f"{fps:4.1f} FPS | hands: {hands_detected} | {async_status} | q/esc to quit"
    cv2.putText(
        frame,
        status,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_webcam(
    args: argparse.Namespace,
    landmarker: vision.HandLandmarker,
    latest_result: LatestHandResult,
) -> None:
    capture = open_camera(args.camera_index, args.width, args.height)
    start_time = time.monotonic()
    previous_frame_time = start_time
    previous_timestamp_ms = -1
    fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            if timestamp_ms <= previous_timestamp_ms:
                timestamp_ms = previous_timestamp_ms + 1
            previous_timestamp_ms = timestamp_ms

            landmarker.detect_async(mp_image, timestamp_ms)

            now = time.monotonic()
            instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instantaneous_fps)
            previous_frame_time = now

            result, result_timestamp_ms = latest_result.get()
            if result is None:
                hands_detected = 0
                result_age_ms = None
            else:
                hands_detected = len(result.hand_landmarks)
                result_age_ms = max(0, timestamp_ms - result_timestamp_ms)
                draw_result(frame, result)

            draw_status(frame, fps, hands_detected, result_age_ms)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def video_inputs(args: argparse.Namespace) -> list[Path]:
    videos = [path.expanduser().resolve() for path in args.video]
    if args.sample_videos:
        data_dir = args.data_dir.expanduser().resolve()
        videos.extend(sorted(data_dir.glob("*.mp4")))
    return videos


def run_video(args: argparse.Namespace, landmarker: vision.HandLandmarker, video_path: Path) -> Path:
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_hands.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            draw_result(frame, result)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    model_path = ensure_model(args.model_path)
    if args.check:
        with create_landmarker(args, model_path, running_mode=vision.RunningMode.VIDEO):
            print(f"MediaPipe Hand Landmarker loaded: {model_path}")
            return

    videos = video_inputs(args)
    if videos:
        with create_landmarker(args, model_path, running_mode=vision.RunningMode.VIDEO) as landmarker:
            for video_path in videos:
                run_video(args, landmarker, video_path)
        return

    latest_result = LatestHandResult()
    with create_landmarker(
        args,
        model_path,
        latest_result.update,
        running_mode=vision.RunningMode.LIVE_STREAM,
    ) as landmarker:
        run_webcam(args, landmarker, latest_result)


if __name__ == "__main__":
    main()
