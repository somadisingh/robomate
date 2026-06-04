"""Build a nerfstudio dataset directory from a Supabase recording bundle.

Inputs (typically downloaded from Supabase Storage):
  - video.mp4
  - depth.bin (8-byte timestamp + W*H float32 metres per frame)
  - poses.jsonl
  - intrinsics.json
  - depth dimensions (width, height, frame_count) carried on the recordings row

Outputs (in ``output_dir``):
  - images/frame_NNNNNN.jpg
  - depths/frame_NNNNNN.png  (16-bit millimetres, optional)
  - sparse_pc.ply
  - transforms.json
  - dataset_summary.json
"""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pillow is only required for the on-disk dataset builder that runs inside
# the nerfstudio Modal image. Tests of the pure-numpy helpers shouldn't need
# it, so we lazy-import it inside the functions that use it.


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_json(cls, payload: dict) -> "Intrinsics":
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
        )


@dataclass(frozen=True)
class DatasetSummary:
    output_dir: str
    image_count: int
    video_frame_count: int
    depth_frame_count: int
    point_count: int
    image_width: int
    image_height: int
    depth_width: int
    depth_height: int
    fps: float


# ─────────────────────────────────────────────────────────────────────────────
#  ffmpeg helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe_video(video_path: Path) -> dict:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(res.stdout)


def _ffprobe_frame_pts(video_path: Path) -> np.ndarray:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(res.stdout)
    pts: list[float] = []
    for frame in payload.get("frames", []):
        timestamp = frame.get("best_effort_timestamp_time")
        if timestamp is not None:
            pts.append(float(timestamp))
    return np.array(pts, dtype=np.float64)


def _decode_frames(video_path: Path, frame_count: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-noautorotate",
            "-i",
            str(video_path),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Depth + pose loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_depth(depth_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, depths) parsed from the recording's depth.bin file."""
    record_size = 8 + width * height * 4
    file_size = depth_path.stat().st_size
    if record_size <= 0 or file_size % record_size != 0:
        raise ValueError(
            f"depth.bin size {file_size} not divisible by record size "
            f"{record_size} (w={width}, h={height})"
        )
    count = file_size // record_size
    times = np.empty(count, dtype=np.float64)
    depths = np.empty((count, height, width), dtype=np.float32)
    with depth_path.open("rb") as f:
        for i in range(count):
            times[i] = struct.unpack("<d", f.read(8))[0]
            frame_bytes = f.read(width * height * 4)
            depths[i] = np.frombuffer(frame_bytes, dtype="<f4").reshape(height, width)
    return times, depths


def load_poses_jsonl(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    pose_t = np.array([r["t"] for r in rows], dtype=np.float64)
    pose_pos = np.array([[r["px"], r["py"], r["pz"]] for r in rows], dtype=np.float64)
    pose_quat = np.array(
        [[r["qx"], r["qy"], r["qz"], r["qw"]] for r in rows], dtype=np.float64
    )
    return pose_t, pose_pos, pose_quat


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def nearest_indices(source_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, query_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_right = np.abs(source_times[right] - query_times) < np.abs(
        source_times[left] - query_times
    )
    return np.where(choose_right, right, left)


# ─────────────────────────────────────────────────────────────────────────────
#  Selection / projection
# ─────────────────────────────────────────────────────────────────────────────

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


def depth_points_with_pixels(
    depth: np.ndarray,
    intr: Intrinsics,
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


def write_depth_png(depth: np.ndarray, output_path: Path) -> None:
    from PIL import Image  # lazy import; only needed in the trainer container

    valid = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm).save(output_path)


def camera_transform_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix(quat_xyzw)
    transform[:3, 3] = position
    return transform


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    frame_step: int = 1
    max_frames: int | None = 120
    jpeg_quality: int = 92
    point_stride: int = 2
    point_cloud_depth_step: int = 1
    max_points: int = 300_000
    write_depths: bool = True


def build_dataset(
    *,
    output_dir: Path,
    video_path: Path,
    depth_path: Path,
    poses_text: str,
    intrinsics: dict,
    depth_width: int,
    depth_height: int,
    config: DatasetConfig | None = None,
) -> DatasetSummary:
    """Materialise a Nerfstudio dataset directory.

    ``video_path`` and ``depth_path`` must be local files (typically written
    after downloading from Supabase Storage). ``poses_text`` is the raw JSONL.
    """
    from PIL import Image  # lazy import; only needed in the trainer container

    cfg = config or DatasetConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    intr = Intrinsics.from_json(intrinsics)
    probe = _ffprobe_video(video_path)
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    raw_w = int(stream["width"])
    raw_h = int(stream["height"])
    nb_frames = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or probe["format"]["duration"])
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    num, _, den = fps_text.partition("/")
    fps = float(num) / float(den or 1.0)

    pose_t, pose_pos, pose_quat = load_poses_jsonl(poses_text)
    depth_t, depths = load_depth(depth_path, depth_width, depth_height)

    pts = _ffprobe_frame_pts(video_path)
    sensor_zero = float(pose_t[0])
    if len(pts):
        frame_t = sensor_zero + pts
    else:
        count = nb_frames or max(1, int(round(duration * fps)))
        frame_t = sensor_zero + np.arange(count, dtype=np.float64) / fps

    image_indices = select_frame_indices(
        len(frame_t), frame_step=cfg.frame_step, max_frames=cfg.max_frames
    )
    point_depth_indices = np.arange(
        0, len(depth_t), max(1, cfg.point_cloud_depth_step), dtype=np.int64
    )
    color_indices = set(int(i) for i in nearest_indices(frame_t, depth_t[point_depth_indices]))
    image_indices_set = set(int(i) for i in image_indices)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    if cfg.write_depths:
        (output_dir / "depths").mkdir(parents=True, exist_ok=True)

    decoder = _decode_frames(video_path, len(frame_t))
    assert decoder.stdout is not None
    frame_bytes = raw_w * raw_h * 3
    color_frames: dict[int, np.ndarray] = {}
    decoded = 0
    try:
        for frame_i in range(len(frame_t)):
            buf = decoder.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(raw_h, raw_w, 3)
            if frame_i in image_indices_set:
                Image.fromarray(raw).save(
                    output_dir / "images" / f"frame_{frame_i:06d}.jpg",
                    quality=cfg.jpeg_quality,
                    subsampling=1,
                )
            if frame_i in color_indices:
                color_frames[frame_i] = raw.copy()
            decoded += 1
    finally:
        decoder.stdout.close()
        rc = decoder.wait()
    if decoded == 0:
        raise RuntimeError("No video frames decoded")
    if rc != 0:
        raise RuntimeError(f"ffmpeg decoder failed with code {rc}")
    missing = sorted(color_indices - set(color_frames))
    if missing:
        raise RuntimeError(f"Missing decoded color frames for indices: {missing[:8]}")

    # transforms.json
    pose_idx = nearest_indices(pose_t, frame_t[image_indices])
    frames: list[dict] = []
    for out_i, frame_i in enumerate(image_indices):
        transform = camera_transform_matrix(
            pose_pos[pose_idx[out_i]], pose_quat[pose_idx[out_i]]
        )
        entry: dict = {
            "file_path": f"images/frame_{int(frame_i):06d}.jpg",
            "transform_matrix": transform.tolist(),
            "timestamp": float(frame_t[frame_i]),
            "video_frame_index": int(frame_i),
            "pose_index": int(pose_idx[out_i]),
        }
        if cfg.write_depths:
            t = float(frame_t[frame_i])
            d_idx = int(nearest_indices(depth_t, np.array([t]))[0])
            depth_frame = depths[d_idx]
            rel = f"depths/frame_{int(frame_i):06d}.png"
            write_depth_png(depth_frame, output_dir / rel)
            entry["depth_file_path"] = rel
            entry["depth_index"] = d_idx
        frames.append(entry)

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

    # sparse colored point cloud (seed)
    cloud_pose_idx = nearest_indices(pose_t, depth_t[point_depth_indices])
    cloud_frame_idx = nearest_indices(frame_t, depth_t[point_depth_indices])
    clouds: list[np.ndarray] = []
    cloud_colors: list[np.ndarray] = []
    for local_i, depth_i in enumerate(point_depth_indices):
        points, pixels = depth_points_with_pixels(
            depths[depth_i], intr, depth_width, depth_height, cfg.point_stride
        )
        if len(points) == 0:
            continue
        raw = color_frames[int(cloud_frame_idx[local_i])]
        R = quat_to_matrix(pose_quat[cloud_pose_idx[local_i]])
        world = points @ R.T + pose_pos[cloud_pose_idx[local_i]]
        clouds.append(world)
        cloud_colors.append(raw[pixels[:, 1], pixels[:, 0]])

    if not clouds:
        raise RuntimeError("No valid LiDAR points for seed point-cloud export")
    cloud = np.vstack(clouds)
    rgb = np.vstack(cloud_colors).astype(np.uint8)
    if cfg.max_points > 0 and len(cloud) > cfg.max_points:
        rng = np.random.default_rng(7)
        keep = rng.choice(len(cloud), size=cfg.max_points, replace=False)
        cloud = cloud[keep]
        rgb = rgb[keep]
    write_ascii_ply(output_dir / "sparse_pc.ply", cloud, rgb)

    summary = DatasetSummary(
        output_dir=str(output_dir),
        image_count=len(frames),
        video_frame_count=len(frame_t),
        depth_frame_count=len(depth_t),
        point_count=int(len(cloud)),
        image_width=intr.width,
        image_height=intr.height,
        depth_width=depth_width,
        depth_height=depth_height,
        fps=fps,
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "image_count": summary.image_count,
                "video_frame_count": summary.video_frame_count,
                "depth_frame_count": summary.depth_frame_count,
                "point_count": summary.point_count,
                "image_width": summary.image_width,
                "image_height": summary.image_height,
                "depth_width": summary.depth_width,
                "depth_height": summary.depth_height,
                "fps": summary.fps,
            },
            indent=2,
        )
    )
    return summary
