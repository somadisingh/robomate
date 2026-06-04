from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import process_capture as pc


ROOT = Path(__file__).resolve().parent
DEFAULT_CAPTURE = pc.DEFAULT_CAPTURE
DEFAULT_OUTPUT = ROOT / "outputs" / "nerfstudio" / "iphone-data-1"


def select_frame_indices(
    frame_count: int, frame_step: int = 1, max_frames: int | None = None
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    indices = np.arange(0, frame_count, frame_step, dtype=np.int64)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        keep = np.round(np.linspace(0, len(indices) - 1, max_frames)).astype(np.int64)
        indices = indices[keep]
    return np.unique(indices)


def camera_transform_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pc.quat_to_matrix(quat_xyzw)
    transform[:3, 3] = position
    return transform


def depth_points_with_pixels(
    depth: np.ndarray,
    intr: pc.Intrinsics,
    depth_w: int,
    depth_h: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:depth_h:stride, 0:depth_w:stride]
    z = depth[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05) & (z < 8.5)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid]

    scale_x = depth_w / intr.width
    scale_y = depth_h / intr.height
    fx = intr.fx * scale_x
    fy = intr.fy * scale_y
    cx = intr.cx * scale_x
    cy = intr.cy * scale_y

    x = (xs - cx) * z / fx
    y = -(ys - cy) * z / fy
    points = np.column_stack([x, y, -z])

    rgb_x = np.round((xs + 0.5) * intr.width / depth_w - 0.5).astype(np.int64)
    rgb_y = np.round((ys + 0.5) * intr.height / depth_h - 0.5).astype(np.int64)
    rgb_x = np.clip(rgb_x, 0, intr.width - 1)
    rgb_y = np.clip(rgb_y, 0, intr.height - 1)
    return points, np.column_stack([rgb_x, rgb_y])


def write_depth_png(depth: np.ndarray, output_path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm).save(output_path)


def write_ascii_ply(output_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def decode_video_frames(
    video_path: Path,
    video_info: pc.VideoInfo,
    frame_count: int,
    image_indices: set[int],
    color_indices: set[int],
    image_dir: Path,
    jpeg_quality: int,
) -> dict[int, np.ndarray]:
    image_dir.mkdir(parents=True, exist_ok=True)
    decoder = pc.start_video_decoder(video_path, frame_count)
    assert decoder.stdout is not None

    raw_w, raw_h = video_info.width, video_info.height
    frame_bytes = raw_w * raw_h * 3
    color_frames: dict[int, np.ndarray] = {}
    decoded = 0
    try:
        for frame_i in range(frame_count):
            buf = decoder.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(raw_h, raw_w, 3)
            if frame_i in image_indices:
                Image.fromarray(raw).save(
                    image_dir / f"frame_{frame_i:06d}.jpg",
                    quality=jpeg_quality,
                    subsampling=1,
                )
            if frame_i in color_indices:
                color_frames[frame_i] = raw.copy()
            decoded += 1
    finally:
        decoder.stdout.close()
        return_code = decoder.wait()
    if decoded == 0:
        raise RuntimeError("No video frames decoded")
    if return_code != 0:
        raise RuntimeError(f"ffmpeg decoder failed with code {return_code}")
    if decoded < frame_count:
        raise RuntimeError(f"Decoded {decoded} frames, expected {frame_count}")
    missing = sorted(color_indices - set(color_frames))
    if missing:
        raise RuntimeError(f"Missing decoded color frames for indices: {missing[:8]}")
    return color_frames


def build_colored_point_cloud(
    output_path: Path,
    intr: pc.Intrinsics,
    depth_t: np.ndarray,
    depths: np.ndarray,
    depth_w: int,
    depth_h: int,
    pose_t: np.ndarray,
    pose_pos: np.ndarray,
    pose_quat: np.ndarray,
    frame_t: np.ndarray,
    color_frames: dict[int, np.ndarray],
    point_stride: int,
    depth_step: int,
    max_points: int,
) -> int:
    depth_indices = np.arange(0, len(depth_t), max(1, depth_step), dtype=np.int64)
    pose_idx = pc.nearest_indices(pose_t, depth_t[depth_indices])
    frame_idx = pc.nearest_indices(frame_t, depth_t[depth_indices])

    clouds: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for local_i, depth_i in enumerate(depth_indices):
        points, pixels = depth_points_with_pixels(
            depths[depth_i], intr, depth_w, depth_h, point_stride
        )
        if len(points) == 0:
            continue
        raw = color_frames[int(frame_idx[local_i])]
        R = pc.quat_to_matrix(pose_quat[pose_idx[local_i]])
        world = points @ R.T + pose_pos[pose_idx[local_i]]
        clouds.append(world)
        colors.append(raw[pixels[:, 1], pixels[:, 0]])

    if not clouds:
        raise RuntimeError("No valid LiDAR points for point-cloud export")
    cloud = np.vstack(clouds)
    rgb = np.vstack(colors).astype(np.uint8)
    if max_points > 0 and len(cloud) > max_points:
        rng = np.random.default_rng(7)
        keep = rng.choice(len(cloud), size=max_points, replace=False)
        cloud = cloud[keep]
        rgb = rgb[keep]
    write_ascii_ply(output_path, cloud, rgb)
    return int(len(cloud))


def write_transforms(
    output_dir: Path,
    intr: pc.Intrinsics,
    image_indices: np.ndarray,
    frame_t: np.ndarray,
    pose_t: np.ndarray,
    pose_pos: np.ndarray,
    pose_quat: np.ndarray,
    depth_t: np.ndarray,
    depths: np.ndarray,
    write_depths: bool,
) -> list[dict]:
    depth_dir = output_dir / "depths"
    pose_idx = pc.nearest_indices(pose_t, frame_t[image_indices])
    frames: list[dict] = []
    for out_i, frame_i in enumerate(image_indices):
        timestamp = float(frame_t[frame_i])
        transform = camera_transform_matrix(
            pose_pos[pose_idx[out_i]], pose_quat[pose_idx[out_i]]
        )
        frame: dict = {
            "file_path": f"images/frame_{int(frame_i):06d}.jpg",
            "transform_matrix": transform.tolist(),
            "timestamp": timestamp,
            "video_frame_index": int(frame_i),
            "pose_index": int(pose_idx[out_i]),
        }
        if write_depths:
            depth, depth_i, depth_age = pc.depth_for_time(timestamp, depth_t, depths)
            depth_rel = f"depths/frame_{int(frame_i):06d}.png"
            write_depth_png(depth, output_dir / depth_rel)
            frame["depth_file_path"] = depth_rel
            frame["depth_index"] = int(depth_i)
            frame["depth_age_s"] = float(depth_age)
        frames.append(frame)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": intr.fx,
        "fl_y": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "w": intr.width,
        "h": intr.height,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "depth_unit_scale_factor": 0.001,
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    (output_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
    if write_depths:
        depth_dir.mkdir(parents=True, exist_ok=True)
    return frames


def export_dataset(args: argparse.Namespace) -> dict:
    capture_dir = args.capture_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pc.require_ffmpeg()
    intr = pc.load_intrinsics(capture_dir / "intrinsics.json")
    video_info = pc.ffprobe_video(capture_dir / "video.mp4")
    poses = pc.load_jsonl(capture_dir / "poses.jsonl")
    imu = pc.load_jsonl(capture_dir / "imu.jsonl")
    depth_t, depths, depth_w, depth_h = pc.load_depth(capture_dir)
    pose_t, pose_pos, pose_quat, *_ = pc.rows_to_arrays(poses, imu)
    frame_t = pc.frame_times_from_video(capture_dir / "video.mp4", video_info, pose_t[0])

    image_indices = select_frame_indices(
        len(frame_t), frame_step=args.frame_step, max_frames=args.max_frames
    )
    point_depth_indices = np.arange(
        0, len(depth_t), max(1, args.point_cloud_depth_step), dtype=np.int64
    )
    color_indices = set(
        int(i) for i in pc.nearest_indices(frame_t, depth_t[point_depth_indices])
    )
    color_frames = decode_video_frames(
        capture_dir / "video.mp4",
        video_info,
        len(frame_t),
        set(int(i) for i in image_indices),
        color_indices,
        output_dir / "images",
        args.jpeg_quality,
    )
    frames = write_transforms(
        output_dir,
        intr,
        image_indices,
        frame_t,
        pose_t,
        pose_pos,
        pose_quat,
        depth_t,
        depths,
        write_depths=args.depth_maps,
    )
    point_count = build_colored_point_cloud(
        output_dir / "sparse_pc.ply",
        intr,
        depth_t,
        depths,
        depth_w,
        depth_h,
        pose_t,
        pose_pos,
        pose_quat,
        frame_t,
        color_frames,
        point_stride=args.point_stride,
        depth_step=args.point_cloud_depth_step,
        max_points=args.max_points,
    )

    summary = {
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir),
        "image_count": len(frames),
        "video_frame_count": len(frame_t),
        "depth_frame_count": len(depth_t),
        "point_count": point_count,
        "image_width": intr.width,
        "image_height": intr.height,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "point_stride": args.point_stride,
        "point_cloud_depth_step": args.point_cloud_depth_step,
        "notes": (
            "Images are decoded with ffmpeg -noautorotate so they stay in the "
            f"{intr.width}x{intr.height} ARKit camera frame matching intrinsics.json."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the iPhone ARKit capture to Nerfstudio format."
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--point-cloud-depth-step", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument(
        "--depth-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write nearest/interpolated LiDAR depth maps as 16-bit millimetre PNGs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= 0:
        args.max_frames = None
    return args


def main() -> None:
    summary = export_dataset(parse_args())
    print(f"wrote Nerfstudio dataset: {summary['output_dir']}")
    print(
        "frames={image_count} depth_frames={depth_frame_count} "
        "points={point_count}".format(**summary)
    )


if __name__ == "__main__":
    main()
