from __future__ import annotations

import argparse
from pathlib import Path

from temporal_action_segmentation.env import load_dotenv, openai_model_from_env
from temporal_action_segmentation.hand_tracking import ensure_model
from temporal_action_segmentation.pipeline import PipelineConfig, process_videos


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "temporal_action_segmentation"


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="VITRA-inspired temporal action segmentation for egocentric videos."
    )
    parser.add_argument("--check", action="store_true", help="Download the MediaPipe model, import video deps, and exit.")

    subparsers = parser.add_subparsers(dest="command")
    process = subparsers.add_parser("process", help="Track hands, propose TAS clips, and export JSONL.")
    process.add_argument("--video", type=Path, action="append", default=[], help="Video file to process. Can be repeated.")
    process.add_argument("--input-dir", type=Path, help="Directory of videos to process.")
    process.add_argument("--glob", default="*.mp4", help="Glob used with --input-dir. Default: *.mp4.")
    process.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Output directory.")
    process.add_argument("--model-path", type=Path, default=None, help="MediaPipe Hand Landmarker .task path.")
    process.add_argument("--target-fps", type=float, default=10.0, help="Tracking sample rate. Default: 10.")
    process.add_argument("--max-hands", type=int, default=2, help="Maximum MediaPipe hands. Default: 2.")
    process.add_argument("--detection-confidence", type=float, default=0.5)
    process.add_argument("--presence-confidence", type=float, default=0.5)
    process.add_argument("--tracking-confidence", type=float, default=0.5)
    process.add_argument("--min-seg-s", type=float, default=0.6)
    process.add_argument("--max-seg-s", type=float, default=6.0)
    process.add_argument("--min-visible-ratio", type=float, default=0.6)
    process.add_argument("--min-motion", type=float, default=0.01)
    process.add_argument("--frames-per-clip", type=int, default=8)
    process.add_argument("--skip-contact-sheets", action="store_true")
    process.add_argument("--no-review", action="store_true")
    process.add_argument("--labeler", choices=("none", "openai"), default="none")
    process.add_argument("--openai-model", default=openai_model_from_env())
    process.add_argument("--max-segments", type=int, help="Debug limit per video.")
    process.add_argument("--merge-same-caption", action="store_true")
    return parser.parse_args()


def _videos_from_args(args: argparse.Namespace) -> list[Path]:
    videos = [path.expanduser().resolve() for path in args.video]
    if args.input_dir:
        videos.extend(sorted(args.input_dir.expanduser().resolve().glob(args.glob)))
    videos = [path for path in videos if path.exists()]
    if not videos:
        raise SystemExit("No videos found. Pass --video or --input-dir.")
    return videos


def main() -> None:
    args = parse_args()
    if args.check:
        model_path = ensure_model()
        import cv2
        import mediapipe as mp

        print(f"MediaPipe import ok: {mp.__version__}")
        print(f"OpenCV import ok: {cv2.__version__}")
        print(f"Hand Landmarker model ready: {model_path}")
        return

    if args.command != "process":
        raise SystemExit("Choose a command, for example: tas process --video path/to/video.mp4")

    model_path = ensure_model(args.model_path)
    config = PipelineConfig(
        output_dir=args.output_dir,
        model_path=model_path,
        target_fps=args.target_fps,
        max_hands=args.max_hands,
        detection_confidence=args.detection_confidence,
        presence_confidence=args.presence_confidence,
        tracking_confidence=args.tracking_confidence,
        min_seg_s=args.min_seg_s,
        max_seg_s=args.max_seg_s,
        min_visible_ratio=args.min_visible_ratio,
        min_motion=args.min_motion,
        frames_per_clip=args.frames_per_clip,
        render_contact_sheets=not args.skip_contact_sheets,
        write_review=not args.no_review,
        labeler=args.labeler,
        openai_model=args.openai_model,
        max_segments=args.max_segments,
        merge_same_caption=args.merge_same_caption,
    )
    process_videos(_videos_from_args(args), config)


if __name__ == "__main__":
    main()
