"""Pure-numpy / no-ffmpeg pieces of backend.splat.dataset."""

import numpy as np

from backend.splat.dataset import (
    Intrinsics,
    camera_transform_matrix,
    nearest_indices,
    quat_to_matrix,
    select_frame_indices,
)


def test_select_frame_indices_default_stride():
    out = select_frame_indices(10, frame_step=1)
    assert list(out) == list(range(10))


def test_select_frame_indices_stride_and_max():
    out = select_frame_indices(100, frame_step=2, max_frames=10)
    assert len(out) == 10
    assert out[0] == 0
    assert out[-1] == 98


def test_nearest_indices_picks_closest():
    src = np.array([0.0, 1.0, 2.0, 3.0])
    queries = np.array([0.4, 0.6, 1.5, 3.1])
    out = nearest_indices(src, queries)
    assert list(out) == [0, 1, 1, 3]


def test_quat_to_matrix_identity():
    R = quat_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(R, np.eye(3))


def test_camera_transform_matrix_packs_position_and_rotation():
    M = camera_transform_matrix(
        position=np.array([1.0, 2.0, 3.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert M.shape == (4, 4)
    assert M[0, 3] == 1.0
    assert M[1, 3] == 2.0
    assert M[2, 3] == 3.0
    assert np.allclose(M[:3, :3], np.eye(3))


def test_intrinsics_from_json_coerces_types():
    intr = Intrinsics.from_json(
        {"width": "640", "height": "480", "fx": "500.1", "fy": "500.2", "cx": "320", "cy": "240"}
    )
    assert intr.width == 640
    assert intr.height == 480
    assert intr.fx == 500.1
    assert intr.cx == 320.0
