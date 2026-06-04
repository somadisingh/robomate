from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporal_action_segmentation.env import DEFAULT_OPENAI_MODEL
from temporal_action_segmentation.hand_tracking import track_video, write_tracks_csv
from temporal_action_segmentation.labelers import make_labeler
from temporal_action_segmentation.render import render_contact_sheet, render_speed_plot
from temporal_action_segmentation.review import write_review_html
from temporal_action_segmentation.segmentation import Segment, speed_minima_segments


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    model_path: Path
    target_fps: float = 10.0
    max_hands: int = 2
    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5
    min_seg_s: float = 0.6
    max_seg_s: float = 6.0
    min_visible_ratio: float = 0.6
    min_motion: float = 0.01
    frames_per_clip: int = 8
    render_contact_sheets: bool = True
    write_review: bool = True
    labeler: str = "none"
    openai_model: str = DEFAULT_OPENAI_MODEL
    max_segments: int | None = None
    merge_same_caption: bool = False


def _segment_record(
    video_id: str,
    video_path: Path,
    segment: Segment,
    contact_sheet_path: Path | None,
    speed_plot_path: Path | None,
    label: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "start_sec": round(segment.start_sec, 4),
        "end_sec": round(segment.end_sec, 4),
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "hand": segment.hand,
        "visible_ratio": round(segment.visible_ratio, 4),
        "motion_score": round(segment.motion_score, 6),
        "mean_speed": round(segment.mean_speed, 6),
        "max_speed": round(segment.max_speed, 6),
        "contact_sheet_path": None if contact_sheet_path is None else str(contact_sheet_path),
        "speed_plot_path": None if speed_plot_path is None else str(speed_plot_path),
        **label,
    }


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return output_path


def _merge_adjacent_same_caption(records: list[dict[str, Any]], max_gap_s: float = 0.4) -> list[dict[str, Any]]:
    if not records:
        return []

    merged: list[dict[str, Any]] = []
    for record in records:
        if (
            merged
            and record["video_id"] == merged[-1]["video_id"]
            and record["hand"] == merged[-1]["hand"]
            and record.get("meaningful_manipulation")
            and merged[-1].get("meaningful_manipulation")
            and record.get("caption") == merged[-1].get("caption")
            and record["start_sec"] - merged[-1]["end_sec"] <= max_gap_s
        ):
            merged[-1]["end_sec"] = record["end_sec"]
            merged[-1]["end_frame"] = record["end_frame"]
            merged[-1]["confidence"] = min(float(merged[-1]["confidence"]), float(record["confidence"]))
            merged[-1]["reason"] = f"Merged adjacent clips with caption {record['caption']!r}."
        else:
            merged.append(record.copy())
    return merged


def process_video(video_path: Path, config: PipelineConfig, labeler) -> list[dict[str, Any]]:
    video_tracks = track_video(
        video_path=video_path,
        model_path=config.model_path,
        target_fps=config.target_fps,
        max_hands=config.max_hands,
        detection_confidence=config.detection_confidence,
        presence_confidence=config.presence_confidence,
        tracking_confidence=config.tracking_confidence,
    )

    tracks_path = config.output_dir / "tracks" / f"{video_tracks.video_id}.csv"
    write_tracks_csv(video_tracks, tracks_path)

    records: list[dict[str, Any]] = []
    for hand, track in video_tracks.tracks.items():
        result = speed_minima_segments(
            track.points,
            fps=track.fps,
            min_seg_s=config.min_seg_s,
            max_seg_s=config.max_seg_s,
            visibility=track.visible,
            hand=hand,
        )
        speed_plot_path = render_speed_plot(
            result.speed,
            result.boundaries,
            config.output_dir / "plots" / f"{video_tracks.video_id}_{hand}_speed.jpg",
        )

        for segment in result.segments:
            if segment.visible_ratio < config.min_visible_ratio:
                continue
            if segment.motion_score < config.min_motion:
                continue
            if config.max_segments is not None and len(records) >= config.max_segments:
                break

            contact_sheet_path = None
            if config.render_contact_sheets:
                contact_sheet_path = (
                    config.output_dir
                    / "contact_sheets"
                    / video_tracks.video_id
                    / f"{segment.hand}_{segment.start_frame:06d}_{segment.end_frame:06d}.jpg"
                )
                render_contact_sheet(
                    video_tracks.video_path,
                    video_tracks.samples,
                    track.points,
                    segment,
                    contact_sheet_path,
                    frames_per_clip=config.frames_per_clip,
                )
            if contact_sheet_path is None and config.labeler != "none":
                raise RuntimeError("OpenAI labelling requires contact sheets. Remove --skip-contact-sheets.")
            label = labeler.label(segment, contact_sheet_path or Path()).as_record()
            records.append(
                _segment_record(
                    video_tracks.video_id,
                    video_tracks.video_path,
                    segment,
                    contact_sheet_path,
                    speed_plot_path,
                    label,
                )
            )
    return records


def process_videos(video_paths: list[Path], config: PipelineConfig) -> list[dict[str, Any]]:
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labeler = make_labeler(config.labeler, config.openai_model, output_dir / "cache")

    all_records: list[dict[str, Any]] = []
    for video_path in video_paths:
        all_records.extend(process_video(video_path, config, labeler))

    all_records.sort(key=lambda item: (item["video_id"], item["hand"], item["start_sec"]))
    if config.merge_same_caption:
        all_records = _merge_adjacent_same_caption(all_records)

    jsonl_path = _write_jsonl(all_records, output_dir / "segments.jsonl")
    if config.write_review:
        review_path = write_review_html(all_records, output_dir / "review.html")
        print(f"Wrote review page: {review_path}")
    print(f"Wrote labels: {jsonl_path}")
    return all_records
