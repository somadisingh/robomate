import unittest

import numpy as np

import process_capture


class TimelineAlignmentTest(unittest.TestCase):
    def test_timeline_uses_video_frame_times_not_pose_row_indices(self):
        frame_t = np.array([100.0, 101.333333, 101.35], dtype=np.float64)
        pose_t = np.array([100.0, 100.583387, 101.333456, 101.350124], dtype=np.float64)
        pose_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [99.0, 99.0, 99.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        pose_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (4, 1))
        imu_t = pose_t.copy()
        imu_accel = np.zeros((4, 3), dtype=np.float64)
        imu_gyro = np.zeros((4, 3), dtype=np.float64)
        imu_quat = pose_quat.copy()
        depth_t = pose_t.copy()
        depths = np.ones((4, 2, 2), dtype=np.float32)

        timeline = process_capture.build_timeline(
            frame_t,
            pose_t,
            pose_pos,
            pose_quat,
            imu_t,
            imu_accel,
            imu_gyro,
            imu_quat,
            depth_t,
            depths,
        )

        np.testing.assert_allclose(timeline.frame_t, frame_t)
        np.testing.assert_allclose(timeline.pose_pos[1], [1.0, 0.0, 0.0])
        self.assertEqual(int(timeline.depth_idx[1]), 2)

    def test_depth_unprojection_uses_arkit_camera_axes(self):
        intr = process_capture.Intrinsics(
            width=2,
            height=2,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        )
        depth = np.ones((2, 2), dtype=np.float32)

        points = process_capture.project_depth_points(
            depth,
            intr,
            depth_w=2,
            depth_h=2,
            stride=1,
        )

        np.testing.assert_allclose(points[0], [0.0, -0.0, -1.0])
        np.testing.assert_allclose(points[-1], [1.0, -1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
