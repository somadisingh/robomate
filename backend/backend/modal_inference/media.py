from __future__ import annotations

from pathlib import Path


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_suffix(suffix: str | None, default: str = ".mp4") -> str:
    if not suffix:
        return default
    clean = suffix.strip().lower()
    if not clean:
        return default
    return clean if clean.startswith(".") else f".{clean}"


def is_video_suffix(suffix: str | None) -> bool:
    return normalize_suffix(suffix) in VIDEO_SUFFIXES


def write_media_bytes(media: bytes, directory: Path, suffix: str | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"input{normalize_suffix(suffix)}"
    path.write_bytes(media)
    return path


def bounded_max_frames(max_frames: int | None) -> int | None:
    if max_frames is None or max_frames <= 0:
        return None
    return max_frames


def output_json(path: Path, payload: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
