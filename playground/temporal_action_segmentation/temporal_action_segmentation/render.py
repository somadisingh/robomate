from __future__ import annotations

from pathlib import Path

import numpy as np

from temporal_action_segmentation.models import VideoSample
from temporal_action_segmentation.segmentation import Segment


HAND_COLORS = {
    "left": (255, 70, 80),
    "right": (40, 180, 255),
}


def _norm_to_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    if not np.isfinite(point).all():
        return None
    x = int(np.clip(point[0], 0.0, 1.0) * max(1, width - 1))
    y = int(np.clip(point[1], 0.0, 1.0) * max(1, height - 1))
    return x, y


def _read_frame(capture, source_frame: int):
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def draw_trajectory(
    frame,
    points: np.ndarray,
    hand: str,
    current_point: np.ndarray | None = None,
) -> None:
    import cv2

    height, width = frame.shape[:2]
    color = HAND_COLORS.get(hand, (80, 255, 120))
    pixels = [
        pixel
        for pixel in (_norm_to_pixel(point, width, height) for point in points)
        if pixel is not None
    ]
    if len(pixels) >= 2:
        for start, end in zip(pixels[:-1], pixels[1:]):
            cv2.line(frame, start, end, color, 4, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(frame, pixel, 4, color, -1, cv2.LINE_AA)

    if current_point is not None:
        current = _norm_to_pixel(current_point, width, height)
        if current is not None:
            cv2.circle(frame, current, 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, current, 7, color, -1, cv2.LINE_AA)


def _put_label(frame, text: str, origin: tuple[int, int]) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def render_contact_sheet(
    video_path: Path,
    samples: list[VideoSample],
    track_points: np.ndarray,
    segment: Segment,
    output_path: Path,
    frames_per_clip: int = 8,
    tile_width: int = 360,
) -> Path:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(segment.start_frame, segment.end_frame, frames_per_clip)
    indices = sorted({int(round(index)) for index in indices})

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for contact sheet: {video_path}")

    tiles = []
    segment_points = track_points[segment.start_frame : segment.end_frame + 1]
    try:
        for index in indices:
            frame = _read_frame(capture, samples[index].source_frame)
            if frame is None:
                continue
            draw_trajectory(frame, segment_points, segment.hand, track_points[index])
            _put_label(
                frame,
                f"{segment.hand.upper()} {samples[index].time_sec:.2f}s",
                (12, 30),
            )
            height, width = frame.shape[:2]
            tile_height = max(1, int(round(tile_width * height / max(width, 1))))
            tiles.append(cv2.resize(frame, (tile_width, tile_height)))
    finally:
        capture.release()

    if not tiles:
        raise RuntimeError(f"Could not render any frames for {video_path}")

    tile_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] < tile_height:
            pad = np.full((tile_height - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.vstack((tile, pad))
        padded_tiles.append(tile)

    columns = min(4, len(padded_tiles))
    rows = int(np.ceil(len(padded_tiles) / columns))
    blank = np.full_like(padded_tiles[0], 245)
    grid_tiles = padded_tiles + [blank] * (rows * columns - len(padded_tiles))
    row_images = [
        np.hstack(grid_tiles[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    sheet = np.vstack(row_images)
    cv2.imwrite(str(output_path), sheet)
    return output_path


def render_speed_plot(
    speed: np.ndarray,
    boundaries: list[int],
    output_path: Path,
    width: int = 900,
    height: int = 240,
) -> Path:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 48, width - 16, 18, height - 36
    cv2.rectangle(canvas, (left, top), (right, bottom), (225, 225, 225), 1)

    if len(speed):
        values = np.asarray(speed, dtype=float)
        scale = max(float(values.max() - values.min()), 1e-9)
        points = []
        for index, value in enumerate(values):
            x = left + int(round(index * (right - left) / max(1, len(values) - 1)))
            y_norm = (float(value) - float(values.min())) / scale
            y = bottom - int(round(y_norm * (bottom - top)))
            points.append((x, y))
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(canvas, start, end, (35, 105, 215), 2, cv2.LINE_AA)

    for boundary in boundaries:
        x = left + int(round(boundary * (right - left) / max(1, len(speed) - 1)))
        cv2.line(canvas, (x, top), (x, bottom), (40, 40, 220), 1, cv2.LINE_AA)

    cv2.putText(canvas, "smoothed hand speed + cuts", (left, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)
    return output_path
