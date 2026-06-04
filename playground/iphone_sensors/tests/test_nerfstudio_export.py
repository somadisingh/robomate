import unittest

import numpy as np

import export_nerfstudio
import process_capture


class NerfstudioExportTest(unittest.TestCase):
    def test_select_frame_indices_applies_stride_then_limit(self):
        indices = export_nerfstudio.select_frame_indices(
            frame_count=10, frame_step=2, max_frames=3
        )
        np.testing.assert_array_equal(indices, np.array([0, 4, 8]))

    def test_camera_transform_matrix_uses_camera_to_world_pose(self):
        transform = export_nerfstudio.camera_transform_matrix(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(transform[:3, :3], np.eye(3))
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_depth_points_with_pixels_match_arkit_axes_and_rgb_scale(self):
        intr = process_capture.Intrinsics(
            width=4,
            height=4,
            fx=2.0,
            fy=2.0,
            cx=0.0,
            cy=0.0,
        )
        depth = np.ones((2, 2), dtype=np.float32)

        points, pixels = export_nerfstudio.depth_points_with_pixels(
            depth,
            intr,
            depth_w=2,
            depth_h=2,
            stride=1,
        )

        np.testing.assert_allclose(points[0], [0.0, -0.0, -1.0])
        np.testing.assert_allclose(points[-1], [1.0, -1.0, -1.0])
        np.testing.assert_array_equal(pixels[0], [0, 0])
        np.testing.assert_array_equal(pixels[-1], [2, 2])


if __name__ == "__main__":
    unittest.main()
