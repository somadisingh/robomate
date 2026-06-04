from __future__ import annotations

import numpy as np

from temporal_action_segmentation.segmentation import (
    interpolate_missing,
    speed_minima_segments,
)


def test_interpolate_missing_fills_internal_gaps_per_dimension() -> None:
    track = np.array(
        [
            [0.0, 0.0],
            [np.nan, np.nan],
            [2.0, 4.0],
            [3.0, np.nan],
        ]
    )

    filled = interpolate_missing(track)

    np.testing.assert_allclose(
        filled,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_speed_minima_segments_cuts_near_motion_pause() -> None:
    fps = 10.0
    first_move = np.column_stack((np.linspace(0.0, 0.45, 20), np.zeros(20)))
    pause = np.column_stack((np.full(12, 0.45), np.zeros(12)))
    second_move = np.column_stack((np.linspace(0.45, 0.9, 20), np.zeros(20)))
    track = np.vstack((first_move, pause, second_move))

    result = speed_minima_segments(track, fps=fps, min_seg_s=1.0, max_seg_s=10.0)

    inner_boundaries = result.boundaries[1:-1]
    assert len(result.segments) == 2
    assert any(18 <= boundary <= 32 for boundary in inner_boundaries)
    assert result.segments[0].start_frame == 0
    assert result.segments[-1].end_frame == len(track) - 1


def test_speed_minima_segments_splits_long_constant_motion_only_by_max_duration() -> None:
    fps = 10.0
    track = np.column_stack((np.linspace(0.0, 1.0, 100), np.zeros(100)))

    result = speed_minima_segments(track, fps=fps, min_seg_s=0.8, max_seg_s=3.0)

    assert all(segment.end_frame - segment.start_frame <= 30 for segment in result.segments)
    assert result.boundaries == [0, 30, 60, 90, 99]
