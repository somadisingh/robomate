from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class YoloTask:
    key: str
    label: str
    model: str
    mode: str


TASKS = {
    "detect": YoloTask("detect", "object_detection", "yolo26n.pt", "predict"),
    "obb": YoloTask("obb", "obb_object_detection", "yolo26n-obb.pt", "predict"),
    "track": YoloTask("track", "object_tracking", "yolo26n.pt", "track"),
    "semantic": YoloTask("semantic", "semantic_segmentation", "yolo26n-sem.pt", "predict"),
    "instance": YoloTask("instance", "instance_segmentation", "yolo26n-seg.pt", "predict"),
}


def playground_path() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    return playground_path() / "data"


def default_output_path() -> Path:
    return playground_path() / "outputs" / "yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLO26 video tasks against playground videos."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_path(),
        help="Directory containing input .mp4 files. Default: ../data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_path(),
        help="Directory for rendered outputs. Default: ../outputs/yolo.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="Specific input video to process. Can be repeated. Defaults to all .mp4 files in --data-dir.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
        help="Tasks to run. Default: all tasks.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device string. Use 'auto' to prefer mps, then cuda, then cpu. Default: auto.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection-style tasks. Default: 0.25.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config for object tracking. Default: bytetrack.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def input_videos(args: argparse.Namespace) -> list[Path]:
    videos = [path.expanduser().resolve() for path in args.video]
    if not videos:
        videos = sorted(args.data_dir.expanduser().resolve().glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found in {args.data_dir}")
    missing = [path for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing video files: " + ", ".join(str(path) for path in missing))
    return videos


def video_metadata(video_path: Path) -> tuple[float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return fps, (width, height)


def rendered_video_metadata(video_path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": round(frame_count / fps, 3) if fps else None,
    }


def output_path(output_dir: Path, task: YoloTask, video_path: Path) -> Path:
    return output_dir / task.label / f"{video_path.stem}_{task.label}.mp4"


def run_task_on_video(
    task: YoloTask,
    video_path: Path,
    output_file: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fps, _input_size = video_metadata(video_path)

    model = YOLO(task.model)
    start = time.monotonic()

    common_kwargs = dict(
        source=str(video_path),
        stream=True,
        device=device,
        imgsz=args.imgsz,
        verbose=False,
    )
    if task.mode == "track":
        results = model.track(
            **common_kwargs,
            conf=args.conf,
            tracker=args.tracker,
            persist=True,
        )
    else:
        results = model.predict(
            **common_kwargs,
            conf=args.conf,
        )

    writer = None
    frame_count = 0
    try:
        for result in results:
            annotated = result.plot()
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_file),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video for writing: {output_file}")
            writer.write(annotated)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.monotonic() - start
    return {
        "task": task.key,
        "label": task.label,
        "model": task.model,
        "video": str(video_path),
        "output": str(output_file),
        "frames": frame_count,
        "seconds": round(elapsed, 3),
        "fps": round(frame_count / elapsed, 3) if elapsed else None,
        "device": device,
        "imgsz": args.imgsz,
        "status": "rendered",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    videos = input_videos(args)
    requested_device = select_device(args.device)

    manifest = []
    for task_key in args.tasks:
        task = TASKS[task_key]
        for video_path in videos:
            out = output_path(output_dir, task, video_path)
            if out.exists() and not args.overwrite:
                print(f"skip existing {out}", flush=True)
                manifest_entry = {
                    "task": task.key,
                    "label": task.label,
                    "model": task.model,
                    "video": str(video_path),
                    "output": str(out),
                    "device": requested_device,
                    "imgsz": args.imgsz,
                    "status": "existing",
                }
                manifest_entry.update(rendered_video_metadata(out))
                manifest.append(manifest_entry)
                continue

            print(
                f"run {task.label} on {video_path.name} with {task.model} device={requested_device}",
                flush=True,
            )
            try:
                manifest.append(run_task_on_video(task, video_path, out, args, requested_device))
            except Exception:
                if args.device != "auto" or requested_device == "cpu":
                    raise
                print(f"{requested_device} failed for {task.label}; retrying on cpu", flush=True)
                manifest.append(run_task_on_video(task, video_path, out, args, "cpu"))

    if manifest:
        manifest_path = output_dir / "manifest.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
