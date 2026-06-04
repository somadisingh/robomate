from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def color_for(label: str) -> tuple[int, int, int]:
    value = sum((index + 1) * ord(char) for index, char in enumerate(label))
    palette = (
        (64, 180, 255),
        (80, 220, 120),
        (255, 160, 60),
        (220, 90, 255),
        (255, 95, 95),
        (80, 210, 220),
        (200, 220, 80),
    )
    return palette[value % len(palette)]


def put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.75,
) -> None:
    x, y = origin
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top_left = (max(0, x - 5), max(0, y - text_height - baseline - 5))
    bottom_right = (min(frame.shape[1] - 1, x + text_width + 5), min(frame.shape[0] - 1, y + baseline + 5))
    cv2.rectangle(frame, top_left, bottom_right, (18, 20, 24), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_header(frame: np.ndarray, title: str, detail: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (18, 20, 24), -1)
    cv2.putText(frame, title, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    detail_size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(
        frame,
        detail,
        (max(18, frame.shape[1] - detail_size[0] - 18), 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 220, 235),
        2,
        cv2.LINE_AA,
    )


def maybe_resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    height = int(round(frame.shape[0] * max_width / frame.shape[1]))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def make_writer(path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def polygon_pixels(instance: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    polygon = instance.get("mask_polygon_xyn")
    if not polygon or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * max(1, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * max(1, height - 1)
    return points.astype(np.int32).reshape((-1, 1, 2))


def draw_instances(frame: np.ndarray, instances: list[dict[str, Any]]) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for instance in instances:
        label = str(instance.get("class_name") or instance.get("label") or "object")
        color = color_for(label)
        polygon = polygon_pixels(instance, width, height)
        if polygon is not None:
            cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    for instance in instances:
        label = str(instance.get("class_name") or instance.get("label") or "object")
        confidence = instance.get("confidence")
        color = color_for(label)
        polygon = polygon_pixels(instance, width, height)
        if polygon is not None:
            cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        box = instance.get("box_xyxy")
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            text = f"{label} {float(confidence):.2f}" if confidence is not None else label
            put_label(frame, text, (x1 + 6, max(26, y1 - 8)), color)


def render_detection_video(
    video_path: Path,
    json_path: Path,
    output_path: Path,
    title: str,
    max_width: int,
) -> Path:
    data = load_json(json_path)
    frames = {int(frame["frame_index"]): frame for frame in data.get("frames", [])}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames))
    writer = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = frames.get(index, {})
            draw_instances(frame, list(result.get("instances", [])))
            draw_header(frame, title, f"frame {index + 1}/{max(total, index + 1)}")
            frame = maybe_resize(frame, max_width)
            writer = writer or make_writer(output_path, fps, frame)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return output_path


def norm_point(point: dict[str, Any], width: int, height: int) -> tuple[int, int]:
    x = float(point.get("x", 0.0))
    y = float(point.get("y", 0.0))
    return (
        int(np.clip(x, 0.0, 1.0) * max(1, width - 1)),
        int(np.clip(y, 0.0, 1.0) * max(1, height - 1)),
    )


def draw_hands(frame: np.ndarray, hands: list[dict[str, Any]]) -> None:
    height, width = frame.shape[:2]
    for hand_index, hand in enumerate(hands):
        handedness = str(hand.get("handedness") or f"hand-{hand_index + 1}")
        confidence = hand.get("confidence")
        color = (80, 210, 255) if handedness.lower() == "right" else (255, 130, 90)
        landmarks = list(hand.get("landmarks", []))
        points = [norm_point(point, width, height) for point in landmarks]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
        if points:
            text = f"{handedness} {float(confidence):.2f}" if confidence is not None else handedness
            put_label(frame, text, (points[0][0] + 8, max(28, points[0][1] - 8)), color)


def read_frame(capture: cv2.VideoCapture, source_frame: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = capture.read()
    return frame if ok else None


def render_hands_video(video_path: Path, json_path: Path, output_path: Path, max_width: int) -> Path:
    data = load_json(json_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(data.get("sample_fps") or data.get("settings", {}).get("target_fps") or 10.0)
    frames = list(data.get("frames", []))
    writer = None
    try:
        for index, result in enumerate(frames):
            source_frame = int(result.get("source_frame", result.get("frame_index", index)))
            frame = read_frame(capture, source_frame)
            if frame is None:
                continue
            draw_hands(frame, list(result.get("hands", [])))
            time_sec = float(result.get("time_sec", index / max(fps, 1e-6)))
            draw_header(frame, "MediaPipe hand landmarks", f"{time_sec:.2f}s")
            frame = maybe_resize(frame, max_width)
            writer = writer or make_writer(output_path, fps, frame)
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return output_path


def segment_bounds(segment: dict[str, Any], fps: float) -> tuple[float, float]:
    start = segment.get("start_time_sec", segment.get("start_sec"))
    end = segment.get("end_time_sec", segment.get("end_sec"))
    if start is not None and end is not None:
        return float(start), float(end)
    start_frame = float(segment.get("start_frame", 0))
    end_frame = float(segment.get("end_frame", start_frame))
    return start_frame / max(fps, 1e-6), end_frame / max(fps, 1e-6)


def segment_label(segment: dict[str, Any]) -> str:
    return str(segment.get("label") or segment.get("class_name") or segment.get("action") or "segment")


def draw_temporal_overlay(
    frame: np.ndarray,
    time_sec: float,
    duration: float,
    segments: list[dict[str, Any]],
    fps: float,
) -> None:
    height, width = frame.shape[:2]
    active = []
    for segment in segments:
        start, end = segment_bounds(segment, fps)
        if start <= time_sec <= end:
            active.append(segment_label(segment))
    status = ", ".join(active) if active else "no temporal segment"
    draw_header(frame, "Temporal action segmentation", f"{time_sec:.2f}s | {status}")

    bar_top = height - 38
    bar_left = 24
    bar_right = width - 24
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_top + 14), (30, 32, 38), -1)
    for segment in segments:
        start, end = segment_bounds(segment, fps)
        x1 = bar_left + int(np.clip(start / max(duration, 1e-6), 0.0, 1.0) * (bar_right - bar_left))
        x2 = bar_left + int(np.clip(end / max(duration, 1e-6), 0.0, 1.0) * (bar_right - bar_left))
        color = color_for(segment_label(segment))
        cv2.rectangle(frame, (x1, bar_top), (max(x1 + 2, x2), bar_top + 14), color, -1)
    progress_x = bar_left + int(np.clip(time_sec / max(duration, 1e-6), 0.0, 1.0) * (bar_right - bar_left))
    cv2.line(frame, (progress_x, bar_top - 8), (progress_x, bar_top + 22), (255, 255, 255), 2, cv2.LINE_AA)


def render_temporal_video(video_path: Path, json_path: Path, output_path: Path, max_width: int) -> Path:
    data = load_json(json_path)
    segments = list(data.get("segments", []))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / max(fps, 1e-6)
    writer = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            draw_temporal_overlay(frame, index / max(fps, 1e-6), duration, segments, fps)
            frame = maybe_resize(frame, max_width)
            writer = writer or make_writer(output_path, fps, frame)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return output_path


def write_index(output_dir: Path, rendered: list[tuple[str, str]], summaries: dict[str, str]) -> Path:
    cards = []
    for title, filename in rendered:
        escaped_title = html.escape(title)
        escaped_filename = html.escape(filename)
        cards.append(
            f"""
            <section>
              <h2>{escaped_title}</h2>
              <video controls preload="metadata" src="{escaped_filename}"></video>
              <p>{html.escape(summaries.get(filename, ""))}</p>
            </section>
            """
        )
    json_links = "\n".join(
        f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
        for path in sorted(output_dir.glob("*.json"))
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video 2 Modal Inference Outputs</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #171a1f; background: #f4f6f8; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ color: #4f5b67; line-height: 1.45; }}
    section {{ margin-top: 20px; padding: 16px; background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; }}
    video {{ display: block; width: 100%; max-height: 70vh; background: #111; border-radius: 6px; }}
    ul {{ padding-left: 20px; }}
    a {{ color: #0757a8; }}
  </style>
</head>
<body>
  <main>
    <h1>Video 2 Modal Inference Outputs</h1>
    <p>Viewable overlays rendered from Modal JSON results for YOLO detection, YOLO instance segmentation, SAM 3.1 instance segmentation, MediaPipe hand landmarks, and temporal action segmentation.</p>
    {''.join(cards)}
    <section>
      <h2>Raw JSON</h2>
      <ul>{json_links}</ul>
    </section>
  </main>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def summarize_counts(output_dir: Path) -> dict[str, str]:
    summaries = {}
    for filename in ("yolo-detect.json", "yolo-instance.json", "sam-instance.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        count = sum(len(frame.get("instances", [])) for frame in data.get("frames", []))
        summaries[filename.replace(".json", ".mp4")] = f"{data.get('frame_count', 0)} frames, {count} total instances."

    hands_path = output_dir / "mediapipe-hands.json"
    if hands_path.exists():
        data = load_json(hands_path)
        detected = sum(1 for frame in data.get("frames", []) if frame.get("hands"))
        summaries["mediapipe-hands.mp4"] = f"{data.get('frame_count', 0)} sampled frames, hands detected in {detected} frames."

    temporal_path = output_dir / "temporal-action-segmentation.json"
    if temporal_path.exists():
        data = load_json(temporal_path)
        summaries["temporal-action-segmentation.mp4"] = f"{data.get('segment_count', 0)} temporal segments detected."
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    output_dir = args.output_dir
    rendered: list[tuple[str, str]] = []

    jobs = (
        ("YOLO26 object detection", "yolo-detect.json", "yolo-detect.mp4"),
        ("YOLO26 instance segmentation", "yolo-instance.json", "yolo-instance.mp4"),
        ("SAM 3.1 instance segmentation", "sam-instance.json", "sam-instance.mp4"),
    )
    for title, source_name, output_name in jobs:
        source_path = output_dir / source_name
        if source_path.exists():
            render_detection_video(args.video, source_path, output_dir / output_name, title, args.max_width)
            rendered.append((title, output_name))

    hands_path = output_dir / "mediapipe-hands.json"
    if hands_path.exists():
        render_hands_video(args.video, hands_path, output_dir / "mediapipe-hands.mp4", args.max_width)
        rendered.append(("MediaPipe hand landmarks", "mediapipe-hands.mp4"))

    temporal_path = output_dir / "temporal-action-segmentation.json"
    if temporal_path.exists():
        render_temporal_video(args.video, temporal_path, output_dir / "temporal-action-segmentation.mp4", args.max_width)
        rendered.append(("Temporal action segmentation", "temporal-action-segmentation.mp4"))

    summaries = summarize_counts(output_dir)
    index_path = write_index(output_dir, rendered, summaries)
    print(f"Wrote {len(rendered)} videos and {index_path}")


if __name__ == "__main__":
    main()
