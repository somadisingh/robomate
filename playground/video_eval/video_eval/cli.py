from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from video_eval.evaluator import (
    DEFAULT_MODEL,
    build_evaluation_prompt,
    evaluate_video,
    load_environment,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    load_environment()

    video_path = args.video.expanduser().resolve()
    model = args.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL

    if args.dry_run:
        payload = {
            "video_path": str(video_path),
            "task_description": args.task,
            "model": model,
            "prompt": build_evaluation_prompt(args.task),
        }
    else:
        result = evaluate_video(
            video_path=video_path,
            task_description=args.task,
            model=model,
            cleanup_uploaded=not args.keep_upload,
        )
        payload = result.model_dump()

    output = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a task succeeded in a video with Gemini."
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to a local video file.")
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Gemini model id. Defaults to GEMINI_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON result.")
    parser.add_argument(
        "--keep-upload",
        action="store_true",
        help="Keep the uploaded Gemini file instead of deleting it after evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved prompt/configuration without uploading the video.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

