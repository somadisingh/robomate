from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    hand: str = ""
    visible_ratio: float = 0.0
    motion_score: float = 0.0
    mean_speed: float = 0.0
    max_speed: float = 0.0


@dataclass(frozen=True)
class SegmentationResult:
    segments: list[Segment]
    boundaries: list[int]
    candidates: list[int]
    speed: np.ndarray
    smoothed_points: np.ndarray


def interpolate_missing(track_xy: np.ndarray) -> np.ndarray:
    """Fill NaNs in a T x D trajectory by linear interpolation per dimension."""
    points = np.asarray(track_xy, dtype=float)
    if points.ndim != 2:
        raise ValueError("track_xy must be a 2D array shaped T x D")
    if len(points) == 0:
        return points.copy()

    filled = points.copy()
    frame_index = np.arange(len(points))
    for dim in range(points.shape[1]):
        values = points[:, dim]
        good = np.isfinite(values)
        if good.sum() == 0:
            filled[:, dim] = 0.0
        elif good.sum() == 1:
            filled[:, dim] = values[good][0]
        else:
            filled[:, dim] = np.interp(frame_index, frame_index[good], values[good])
    return filled


def gaussian_smooth(values: np.ndarray, sigma: float, axis: int = 0) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if sigma <= 0 or data.size == 0:
        return data.copy()

    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    moved = np.moveaxis(data, axis, 0)
    flat = moved.reshape((moved.shape[0], -1))
    smoothed = np.empty_like(flat, dtype=float)
    for column in range(flat.shape[1]):
        padded = np.pad(flat[:, column], radius, mode="edge")
        smoothed[:, column] = np.convolve(padded, kernel, mode="valid")
    return np.moveaxis(smoothed.reshape(moved.shape), 0, axis)


def hand_speed(track_xy: np.ndarray, fps: float, sigma_s: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    filled = interpolate_missing(track_xy)
    sigma_frames = max(1.0, sigma_s * fps)
    smoothed_points = gaussian_smooth(filled, sigma=sigma_frames, axis=0)
    frame_delta = np.diff(smoothed_points, axis=0, prepend=smoothed_points[:1])
    speed = np.linalg.norm(frame_delta, axis=1) * fps
    speed = gaussian_smooth(speed, sigma=sigma_frames, axis=0)
    return speed, smoothed_points


def _collapse_candidate_runs(candidates: list[int], speed: np.ndarray, max_gap: int) -> list[int]:
    if not candidates:
        return []

    runs: list[list[int]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate - runs[-1][-1] <= max_gap:
            runs[-1].append(candidate)
        else:
            runs.append([candidate])

    collapsed = []
    for run in runs:
        run_speeds = speed[run]
        lowest = float(run_speeds.min())
        best = [frame for frame in run if abs(float(speed[frame]) - lowest) <= 1e-12]
        collapsed.append(best[len(best) // 2])
    return collapsed


def speed_minima_candidates(
    speed: np.ndarray,
    fps: float,
    window_s: float = 0.5,
    speed_percentile: float = 40.0,
    prominence_ratio: float = 0.02,
) -> list[int]:
    if len(speed) < 3:
        return []

    half_window = max(1, int(round(window_s * fps / 2.0)))
    low_speed_threshold = float(np.percentile(speed, speed_percentile))
    dynamic_range = float(np.max(speed) - np.min(speed))
    if dynamic_range <= 1e-9:
        return []

    min_prominence = max(1e-6, dynamic_range * prominence_ratio)
    low_speed = speed <= low_speed_threshold
    candidates: list[int] = []
    index = 0
    while index < len(speed):
        if not low_speed[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(speed) and low_speed[index + 1]:
            index += 1
        end = index
        index += 1

        if start <= half_window or end >= len(speed) - half_window - 1:
            continue
        run = np.arange(start, end + 1)
        lowest = float(speed[run].min())
        best = [int(frame) for frame in run if abs(float(speed[frame]) - lowest) <= 1e-12]
        candidate = best[len(best) // 2]
        context_start = max(0, start - half_window)
        context_end = min(len(speed), end + half_window + 1)
        if float(speed[context_start:context_end].max()) - lowest >= min_prominence:
            candidates.append(candidate)

    return _collapse_candidate_runs(candidates, speed, max_gap=half_window)


def _final_boundaries(
    candidates: list[int],
    total_frames: int,
    fps: float,
    min_seg_s: float,
    max_seg_s: float,
) -> list[int]:
    if total_frames <= 0:
        return []

    min_len = max(1, int(round(min_seg_s * fps)))
    max_len = max(min_len, int(round(max_seg_s * fps)))
    end = total_frames - 1

    proposed = [0]
    for candidate in candidates:
        if candidate - proposed[-1] >= min_len and end - candidate >= min_len:
            proposed.append(candidate)
    if end > proposed[-1]:
        proposed.append(end)

    boundaries = [proposed[0]]
    for boundary in proposed[1:]:
        while boundary - boundaries[-1] > max_len:
            boundaries.append(boundaries[-1] + max_len)
        if boundary > boundaries[-1]:
            boundaries.append(boundary)
    return boundaries


def _segment_metrics(
    start: int,
    end: int,
    fps: float,
    hand: str,
    visibility: np.ndarray,
    speed: np.ndarray,
    smoothed_points: np.ndarray,
) -> Segment:
    segment_visibility = visibility[start : end + 1]
    visible_ratio = float(segment_visibility.mean()) if len(segment_visibility) else 0.0
    segment_speed = speed[start : end + 1]
    mean_speed = float(segment_speed.mean()) if len(segment_speed) else 0.0
    max_speed = float(segment_speed.max()) if len(segment_speed) else 0.0

    points = smoothed_points[start : end + 1]
    if len(points) > 1:
        motion_score = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    else:
        motion_score = 0.0

    return Segment(
        start_frame=start,
        end_frame=end,
        start_sec=start / fps,
        end_sec=end / fps,
        hand=hand,
        visible_ratio=visible_ratio,
        motion_score=motion_score,
        mean_speed=mean_speed,
        max_speed=max_speed,
    )


def speed_minima_segments(
    track_xy: np.ndarray,
    fps: float,
    min_seg_s: float = 0.6,
    max_seg_s: float = 6.0,
    window_s: float = 0.5,
    visibility: np.ndarray | None = None,
    hand: str = "",
) -> SegmentationResult:
    """Segment a hand trajectory at local minima in smoothed hand speed."""
    points = np.asarray(track_xy, dtype=float)
    if points.ndim != 2:
        raise ValueError("track_xy must be a 2D array shaped T x D")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_seg_s < min_seg_s:
        raise ValueError("max_seg_s must be greater than or equal to min_seg_s")

    speed, smoothed_points = hand_speed(points, fps=fps)
    candidates = speed_minima_candidates(speed, fps=fps, window_s=window_s)
    boundaries = _final_boundaries(candidates, len(points), fps, min_seg_s, max_seg_s)

    if visibility is None:
        visible = np.isfinite(points).all(axis=1)
    else:
        visible = np.asarray(visibility, dtype=bool)
        if visible.shape != (len(points),):
            raise ValueError("visibility must have shape T")

    segments = [
        _segment_metrics(start, end, fps, hand, visible, speed, smoothed_points)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]
    return SegmentationResult(
        segments=segments,
        boundaries=boundaries,
        candidates=candidates,
        speed=speed,
        smoothed_points=smoothed_points,
    )
