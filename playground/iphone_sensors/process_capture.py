from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_CAPTURE = ROOT.parent / "data" / "iphone-data-1"
DEFAULT_OUTPUT = ROOT / "outputs"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    nb_frames: int
    duration_s: float
    fps_fraction: Fraction
    fps_text: str
    rotation_deg: float


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Timeline:
    frame_t: np.ndarray
    rel_t: np.ndarray
    pose_pos: np.ndarray
    pose_quat: np.ndarray
    imu_idx: np.ndarray
    imu_accel: np.ndarray
    imu_gyro: np.ndarray
    imu_quat: np.ndarray
    depth_idx: np.ndarray
    depth_age_s: np.ndarray
    depth_min: np.ndarray
    depth_p10: np.ndarray
    depth_median: np.ndarray
    depth_p90: np.ndarray
    depth_max: np.ndarray
    pose_euler_deg: np.ndarray
    imu_euler_deg: np.ndarray


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def require_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [binary, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise SystemExit(f"{binary} is required on PATH") from exc


def ffprobe_video(video_path: Path) -> VideoInfo:
    probe = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ]
    )
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    nb_frames = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or probe["format"]["duration"])
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    fps_fraction = Fraction(fps_text)
    rotation = 0.0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
            break
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        nb_frames=nb_frames,
        duration_s=duration,
        fps_fraction=fps_fraction,
        fps_text=fps_text,
        rotation_deg=rotation,
    )


def ffprobe_frame_pts(video_path: Path) -> np.ndarray:
    probe = run_json(
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
        ]
    )
    pts: list[float] = []
    for frame in probe.get("frames", []):
        timestamp = frame.get("best_effort_timestamp_time")
        if timestamp is not None:
            pts.append(float(timestamp))
    return np.array(pts, dtype=np.float64)


def load_intrinsics(path: Path) -> Intrinsics:
    data = json.loads(path.read_text())
    return Intrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_depth(capture_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    depth_meta = metadata["depth"]
    width = int(depth_meta["width"])
    height = int(depth_meta["height"])
    record_size = 8 + width * height * 4
    depth_path = capture_dir / depth_meta["file"]
    file_size = depth_path.stat().st_size
    if file_size % record_size != 0:
        raise ValueError(
            f"Depth file size {file_size} is not divisible by record size {record_size}"
        )
    count = file_size // record_size
    times = np.empty(count, dtype=np.float64)
    depths = np.empty((count, height, width), dtype=np.float32)
    with depth_path.open("rb") as f:
        for i in range(count):
            times[i] = struct.unpack("<d", f.read(8))[0]
            frame_bytes = f.read(width * height * 4)
            depths[i] = np.frombuffer(frame_bytes, dtype="<f4").reshape(height, width)
    return times, depths, width, height


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


def quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def nearest_indices(source_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, query_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    choose_right = np.abs(source_times[right] - query_times) < np.abs(
        source_times[left] - query_times
    )
    return np.where(choose_right, right, left)


def depth_for_time(
    t: float,
    depth_times: np.ndarray,
    depths: np.ndarray,
    max_interp_gap_s: float = 0.25,
) -> tuple[np.ndarray, int, float]:
    right = int(np.searchsorted(depth_times, t, side="left"))
    if right <= 0:
        idx = 0
        return depths[idx], idx, abs(float(depth_times[idx] - t))
    if right >= len(depth_times):
        idx = len(depth_times) - 1
        return depths[idx], idx, abs(float(depth_times[idx] - t))

    left = right - 1
    left_dt = abs(float(t - depth_times[left]))
    right_dt = abs(float(depth_times[right] - t))
    nearest = left if left_dt <= right_dt else right
    age = min(left_dt, right_dt)
    gap = float(depth_times[right] - depth_times[left])
    if gap <= 0.0 or gap > max_interp_gap_s:
        return depths[nearest], nearest, age
    alpha = float((t - depth_times[left]) / gap)
    interpolated = (1.0 - alpha) * depths[left] + alpha * depths[right]
    return interpolated.astype(np.float32), nearest, age


def rows_to_arrays(
    poses: list[dict], imu: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose_t = np.array([r["t"] for r in poses], dtype=np.float64)
    pose_pos = np.array([[r["px"], r["py"], r["pz"]] for r in poses], dtype=np.float64)
    pose_quat = np.array(
        [[r["qx"], r["qy"], r["qz"], r["qw"]] for r in poses], dtype=np.float64
    )
    imu_t = np.array([r["t"] for r in imu], dtype=np.float64)
    imu_accel = np.array([[r["ax"], r["ay"], r["az"]] for r in imu], dtype=np.float64)
    imu_gyro = np.array([[r["gx"], r["gy"], r["gz"]] for r in imu], dtype=np.float64)
    imu_quat = np.array(
        [[r["qx"], r["qy"], r["qz"], r["qw"]] for r in imu], dtype=np.float64
    )
    return pose_t, pose_pos, pose_quat, imu_t, imu_accel, imu_gyro, imu_quat


def build_timeline(
    frame_t: np.ndarray,
    pose_t: np.ndarray,
    pose_pos: np.ndarray,
    pose_quat: np.ndarray,
    imu_t: np.ndarray,
    imu_accel: np.ndarray,
    imu_gyro: np.ndarray,
    imu_quat: np.ndarray,
    depth_t: np.ndarray,
    depths: np.ndarray,
) -> Timeline:
    frame_t = np.asarray(frame_t, dtype=np.float64)
    frame_count = len(frame_t)
    rel_t = frame_t - frame_t[0]
    pose_idx = nearest_indices(pose_t, frame_t)
    imu_idx = nearest_indices(imu_t, frame_t)
    depth_idx = nearest_indices(depth_t, frame_t)
    depth_age = np.abs(depth_t[depth_idx] - frame_t)

    depth_min = np.empty(frame_count)
    depth_p10 = np.empty(frame_count)
    depth_median = np.empty(frame_count)
    depth_p90 = np.empty(frame_count)
    depth_max = np.empty(frame_count)
    for i, t in enumerate(frame_t):
        depth, _, age = depth_for_time(float(t), depth_t, depths)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        depth_min[i] = float(np.min(valid))
        depth_p10[i] = float(np.percentile(valid, 10))
        depth_median[i] = float(np.median(valid))
        depth_p90[i] = float(np.percentile(valid, 90))
        depth_max[i] = float(np.max(valid))
        depth_age[i] = age

    frame_pose_pos = pose_pos[pose_idx]
    frame_pose_quat = pose_quat[pose_idx]
    pose_euler = np.array([quat_to_euler_deg(q) for q in frame_pose_quat])
    imu_euler = np.array([quat_to_euler_deg(q) for q in imu_quat[imu_idx]])

    return Timeline(
        frame_t=frame_t,
        rel_t=rel_t,
        pose_pos=frame_pose_pos,
        pose_quat=frame_pose_quat,
        imu_idx=imu_idx,
        imu_accel=imu_accel[imu_idx],
        imu_gyro=imu_gyro[imu_idx],
        imu_quat=imu_quat[imu_idx],
        depth_idx=depth_idx,
        depth_age_s=depth_age,
        depth_min=depth_min,
        depth_p10=depth_p10,
        depth_median=depth_median,
        depth_p90=depth_p90,
        depth_max=depth_max,
        pose_euler_deg=pose_euler,
        imu_euler_deg=imu_euler,
    )


def frame_times_from_video(
    video_path: Path, video_info: VideoInfo, sensor_zero_t: float
) -> np.ndarray:
    pts = ffprobe_frame_pts(video_path)
    if len(pts):
        return sensor_zero_t + pts
    fps = float(video_info.fps_fraction)
    count = video_info.nb_frames or max(1, int(round(video_info.duration_s * fps)))
    return sensor_zero_t + np.arange(count, dtype=np.float64) / fps


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(depth.astype(np.float32))
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def make_depth_overlay_raw(
    raw_rgb: np.ndarray,
    depth_low: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float = 0.44,
) -> tuple[np.ndarray, np.ndarray]:
    raw_h, raw_w = raw_rgb.shape[:2]
    dense = resize_depth(depth_low, (raw_w, raw_h))
    valid = np.isfinite(dense) & (dense > 0)
    norm = np.clip((dense - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    color = (cm.turbo(norm)[..., :3] * 255.0).astype(np.uint8)
    overlay = raw_rgb.copy()
    blended = (
        (1.0 - alpha) * raw_rgb[valid].astype(np.float32)
        + alpha * color[valid].astype(np.float32)
    )
    overlay[valid] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay, dense


def rotate_to_display(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.ROTATE_270)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate_overlay(
    img: Image.Image,
    t_rel: float,
    depth_median: float,
    depth_age: float,
    vmin: float,
    vmax: float,
) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(30)
    small = load_font(22)
    text = f"t={t_rel:0.2f}s  median depth={depth_median:0.2f}m  depth age={depth_age * 1000:0.0f}ms"
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 16
    draw.rounded_rectangle(
        (24, 24, 24 + bbox[2] + 2 * pad, 24 + bbox[3] + 2 * pad),
        radius=10,
        fill=(0, 0, 0, 132),
    )
    draw.text((24 + pad, 24 + pad), text, fill=(255, 255, 255, 238), font=font)

    bar_w = 34
    bar_h = 280
    x0 = img.width - 66
    y0 = 90
    for j in range(bar_h):
        u = 1.0 - j / max(bar_h - 1, 1)
        color = tuple(int(c * 255) for c in cm.turbo(u)[:3])
        draw.line((x0, y0 + j, x0 + bar_w, y0 + j), fill=color + (230,))
    draw.rectangle((x0, y0, x0 + bar_w, y0 + bar_h), outline=(255, 255, 255, 190), width=2)
    draw.text((x0 - 8, y0 - 28), f"{vmax:0.1f}m", fill=(255, 255, 255, 230), font=small)
    draw.text((x0 - 8, y0 + bar_h + 6), f"{vmin:0.1f}m", fill=(255, 255, 255, 230), font=small)
    return img


def start_raw_video_encoder(
    output_path: Path,
    size: tuple[int, int],
    fps: str,
    crf: int = 22,
) -> subprocess.Popen:
    width, height = size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def start_video_decoder(video_path: Path, frame_count: int) -> subprocess.Popen:
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


def write_frame(process: subprocess.Popen, image: Image.Image) -> None:
    assert process.stdin is not None
    process.stdin.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())


def close_encoder(process: subprocess.Popen, label: str) -> None:
    assert process.stdin is not None
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} ffmpeg encoder failed with code {return_code}")


def draw_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = rect
    font = load_font(18)
    draw.rectangle(rect, outline=(190, 196, 203), width=1)
    draw.text((x0 + 8, y0 + 6), title, fill=(28, 34, 42), font=font)
    draw.line((x0 + 42, y1 - 28, x1 - 12, y1 - 28), fill=(130, 138, 148), width=1)
    draw.line((x0 + 42, y0 + 32, x0 + 42, y1 - 28), fill=(130, 138, 148), width=1)


def scale_points(
    xs: np.ndarray,
    ys: np.ndarray,
    rect: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = rect
    left, top, right, bottom = x0 + 42, y0 + 32, x1 - 12, y1 - 28
    xden = max(xlim[1] - xlim[0], 1e-9)
    yden = max(ylim[1] - ylim[0], 1e-9)
    xp = left + (xs - xlim[0]) / xden * (right - left)
    yp = bottom - (ys - ylim[0]) / yden * (bottom - top)
    return list(zip(np.round(xp).astype(int), np.round(yp).astype(int)))


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    t: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    current_i: int,
    y_label: str = "",
) -> None:
    draw_axes(draw, rect, title)
    x0, y0, x1, y1 = rect
    font = load_font(15)
    all_y = np.concatenate([s[1] for s in series])
    ymin = float(np.nanmin(all_y))
    ymax = float(np.nanmax(all_y))
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)
    xlim = (float(t[0]), float(t[-1]))

    for label, values, color in series:
        pts = scale_points(t[: current_i + 1], values[: current_i + 1], rect, xlim, ylim)
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)

    current_x = scale_points(
        np.array([t[current_i]]),
        np.array([ylim[0]]),
        rect,
        xlim,
        ylim,
    )[0][0]
    draw.line((current_x, y0 + 32, current_x, y1 - 28), fill=(31, 41, 55), width=1)
    draw.text((x0 + 8, y1 - 23), f"{ylim[0]:0.2f}", fill=(84, 94, 108), font=font)
    draw.text((x0 + 8, y0 + 32), f"{ylim[1]:0.2f}", fill=(84, 94, 108), font=font)
    draw.text((x1 - 76, y1 - 23), f"{t[current_i]:0.1f}s", fill=(84, 94, 108), font=font)
    if y_label:
        draw.text((x0 + 8, y0 + 52), y_label, fill=(84, 94, 108), font=font)
    lx = x0 + 110
    for label, _, color in series:
        draw.line((lx, y0 + 18, lx + 22, y0 + 18), fill=color, width=4)
        draw.text((lx + 28, y0 + 9), label, fill=(28, 34, 42), font=font)
        lx += 92


def draw_trajectory(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    timeline: Timeline,
    current_i: int,
) -> None:
    draw_axes(draw, rect, "camera trajectory x/z")
    x = timeline.pose_pos[:, 0]
    z = timeline.pose_pos[:, 2]
    xpad = max(0.05, 0.12 * float(np.ptp(x)))
    zpad = max(0.05, 0.12 * float(np.ptp(z)))
    pts = scale_points(
        x[: current_i + 1],
        z[: current_i + 1],
        rect,
        (float(np.min(x) - xpad), float(np.max(x) + xpad)),
        (float(np.min(z) - zpad), float(np.max(z) + zpad)),
    )
    if len(pts) > 1:
        draw.line(pts, fill=(14, 116, 144), width=4)
    cx, cy = pts[-1]
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(220, 38, 38))
    font = load_font(15)
    x0, y0, x1, y1 = rect
    draw.text(
        (x0 + 10, y1 - 23),
        f"x={x[current_i]:0.2f}m z={z[current_i]:0.2f}m",
        fill=(84, 94, 108),
        font=font,
    )


def make_fusion_panel(
    overlay: Image.Image,
    timeline: Timeline,
    frame_i: int,
    output_size: tuple[int, int] = (1280, 720),
) -> Image.Image:
    width, height = output_size
    canvas = Image.new("RGB", output_size, (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    small = load_font(16)

    video_h = height - 40
    video_w = int(video_h * overlay.width / overlay.height)
    video = overlay.resize((video_w, video_h), Image.Resampling.LANCZOS)
    canvas.paste(video, (20, 20))
    draw.rectangle((20, 20, 20 + video_w, 20 + video_h), outline=(29, 38, 48), width=1)

    x0 = 40 + video_w
    draw.text(
        (x0, 18),
        "iPhone sensor fusion",
        fill=(28, 34, 42),
        font=title_font,
    )
    draw.text(
        (x0, 48),
        f"frame {frame_i + 1}/{len(timeline.rel_t)}  t={timeline.rel_t[frame_i]:0.2f}s",
        fill=(84, 94, 108),
        font=small,
    )

    plot_left = x0
    plot_right = width - 20
    top = 80
    gap = 14
    plot_w = plot_right - plot_left
    plot_h = (height - top - 24 - gap) // 2
    rect1 = (plot_left, top, plot_left + plot_w // 2 - gap // 2, top + plot_h)
    rect2 = (plot_left + plot_w // 2 + gap // 2, top, plot_right, top + plot_h)
    rect3 = (
        plot_left,
        top + plot_h + gap,
        plot_left + plot_w // 2 - gap // 2,
        top + plot_h * 2 + gap,
    )
    rect4 = (
        plot_left + plot_w // 2 + gap // 2,
        top + plot_h + gap,
        plot_right,
        top + plot_h * 2 + gap,
    )

    accel_mag = np.linalg.norm(timeline.imu_accel, axis=1)
    gyro_mag = np.linalg.norm(timeline.imu_gyro, axis=1)
    draw_trajectory(draw, rect1, timeline, frame_i)
    draw_line_chart(
        draw,
        rect2,
        "IMU magnitudes",
        timeline.rel_t,
        [
            ("acc", accel_mag, (17, 94, 89)),
            ("gyro", gyro_mag, (185, 89, 36)),
        ],
        frame_i,
    )
    draw_line_chart(
        draw,
        rect3,
        "depth envelope",
        timeline.rel_t,
        [
            ("p10", timeline.depth_p10, (77, 124, 199)),
            ("med", timeline.depth_median, (34, 139, 34)),
            ("p90", timeline.depth_p90, (175, 94, 156)),
        ],
        frame_i,
        y_label="metres",
    )
    draw_line_chart(
        draw,
        rect4,
        "pose orientation",
        timeline.rel_t,
        [
            ("roll", timeline.pose_euler_deg[:, 0], (36, 96, 160)),
            ("pitch", timeline.pose_euler_deg[:, 1], (180, 83, 9)),
            ("yaw", timeline.pose_euler_deg[:, 2], (111, 66, 193)),
        ],
        frame_i,
        y_label="degrees",
    )
    return canvas


def save_timeline_csv(output_path: Path, timeline: Timeline) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "t",
                "rel_t",
                "depth_idx",
                "depth_age_s",
                "depth_min_m",
                "depth_p10_m",
                "depth_median_m",
                "depth_p90_m",
                "depth_max_m",
                "pose_px_m",
                "pose_py_m",
                "pose_pz_m",
                "pose_qx",
                "pose_qy",
                "pose_qz",
                "pose_qw",
                "imu_idx",
                "imu_ax",
                "imu_ay",
                "imu_az",
                "imu_gx",
                "imu_gy",
                "imu_gz",
                "imu_qx",
                "imu_qy",
                "imu_qz",
                "imu_qw",
            ]
        )
        for i in range(len(timeline.frame_t)):
            writer.writerow(
                [
                    i,
                    timeline.frame_t[i],
                    timeline.rel_t[i],
                    int(timeline.depth_idx[i]),
                    timeline.depth_age_s[i],
                    timeline.depth_min[i],
                    timeline.depth_p10[i],
                    timeline.depth_median[i],
                    timeline.depth_p90[i],
                    timeline.depth_max[i],
                    *timeline.pose_pos[i],
                    *timeline.pose_quat[i],
                    int(timeline.imu_idx[i]),
                    *timeline.imu_accel[i],
                    *timeline.imu_gyro[i],
                    *timeline.imu_quat[i],
                ]
            )


def make_sample_grid(samples: list[tuple[int, Image.Image, Image.Image, Image.Image]], output_path: Path) -> None:
    if not samples:
        return
    thumb_w = 300
    thumb_h = 400
    label_h = 30
    cols = 3
    rows = len(samples)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    headings = ["video", "dense depth", "overlay"]
    for c, heading in enumerate(headings):
        draw.text((c * thumb_w + 12, 6), heading, fill=(30, 37, 48), font=font)
    for r, (frame_i, video, depth, overlay) in enumerate(samples):
        y = r * (thumb_h + label_h) + label_h
        for c, img in enumerate((video, depth, overlay)):
            thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (c * thumb_w, y))
        draw.text((12, y + 8), f"frame {frame_i}", fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_dashboard(output_path: Path, timeline: Timeline, depth_t: np.ndarray) -> None:
    accel_mag = np.linalg.norm(timeline.imu_accel, axis=1)
    gyro_mag = np.linalg.norm(timeline.imu_gyro, axis=1)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    p = timeline.pose_pos
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color="#0e7490", linewidth=2)
    ax3d.scatter(p[0, 0], p[0, 1], p[0, 2], color="#16a34a", label="start")
    ax3d.scatter(p[-1, 0], p[-1, 1], p[-1, 2], color="#dc2626", label="end")
    ax3d.set_title("Camera pose trajectory")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend(loc="upper left")

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(timeline.rel_t, accel_mag, label="accel magnitude", color="#115e59")
    ax.plot(timeline.rel_t, gyro_mag, label="gyro magnitude", color="#b45309")
    ax.set_title("Nearest IMU sample per video frame")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(grid[0, 2])
    ax.fill_between(
        timeline.rel_t,
        timeline.depth_p10,
        timeline.depth_p90,
        color="#93c5fd",
        alpha=0.35,
        label="p10-p90",
    )
    ax.plot(timeline.rel_t, timeline.depth_median, color="#15803d", label="median")
    ax.scatter(depth_t - timeline.frame_t[0], np.full_like(depth_t, timeline.depth_min.min()), s=12, color="#1d4ed8", label="depth frames")
    ax.set_title("LiDAR depth over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("metres")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(grid[1, 1])
    labels = ["roll", "pitch", "yaw"]
    colors = ["#2563eb", "#b45309", "#7c3aed"]
    for i, label in enumerate(labels):
        ax.plot(timeline.rel_t, timeline.pose_euler_deg[:, i], label=f"pose {label}", color=colors[i])
    ax.set_title("Camera pose orientation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(grid[1, 2])
    for i, label in enumerate(labels):
        ax.plot(timeline.rel_t, timeline.imu_euler_deg[:, i], label=f"imu {label}", color=colors[i])
    ax.set_title("IMU attitude orientation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle("iPhone video, LiDAR, pose, and IMU fusion summary", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def project_depth_points(
    depth: np.ndarray,
    intr: Intrinsics,
    depth_w: int,
    depth_h: int,
    stride: int = 4,
) -> np.ndarray:
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
    return np.column_stack([x, y, -z])


def make_world_point_cloud(
    output_png: Path,
    output_ply: Path,
    intr: Intrinsics,
    depth_t: np.ndarray,
    depths: np.ndarray,
    depth_w: int,
    depth_h: int,
    pose_t: np.ndarray,
    pose_pos: np.ndarray,
    pose_quat: np.ndarray,
    max_points: int = 120_000,
) -> None:
    pose_idx = nearest_indices(pose_t, depth_t)
    clouds = []
    colors = []
    for i, d in enumerate(depths):
        pts = project_depth_points(d, intr, depth_w, depth_h, stride=4)
        if len(pts) == 0:
            continue
        R = quat_to_matrix(pose_quat[pose_idx[i]])
        world = pts @ R.T + pose_pos[pose_idx[i]]
        clouds.append(world)
        c = np.full((len(world), 1), i / max(len(depths) - 1, 1), dtype=np.float64)
        colors.append(c)
    if not clouds:
        return
    cloud = np.vstack(clouds)
    color_t = np.vstack(colors).ravel()
    if len(cloud) > max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(cloud), size=max_points, replace=False)
        cloud = cloud[idx]
        color_t = color_t[idx]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(cloud[:, 0], cloud[:, 2], c=color_t, cmap="viridis", s=0.35, alpha=0.55)
    ax.plot(pose_pos[:, 0], pose_pos[:, 2], color="#dc2626", linewidth=2)
    ax.set_title("World point cloud top-down x/z")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=color_t, cmap="viridis", s=0.25, alpha=0.45)
    ax.plot(pose_pos[:, 0], pose_pos[:, 1], pose_pos[:, 2], color="#dc2626", linewidth=2)
    ax.set_title("World point cloud 3D")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    rgb = (cm.viridis(color_t)[:, :3] * 255).astype(np.uint8)
    with output_ply.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cloud)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, color in zip(cloud, rgb):
            f.write(
                f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def make_report(output_dir: Path, summary: dict) -> None:
    rows = "\n".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in summary.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>iPhone Sensor Fusion Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f7f7f4; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 34px; }}
    h2 {{ margin: 32px 0 12px; font-size: 22px; }}
    p {{ color: #4b5563; }}
    video, img {{ width: 100%; height: auto; background: #111827; border: 1px solid #d1d5db; }}
    .videos {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; align-items: start; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 12px; text-align: left; }}
    th {{ width: 240px; color: #374151; }}
    a {{ color: #075985; }}
  </style>
</head>
<body>
<main>
  <h1>iPhone Sensor Fusion Report</h1>
  <p>LiDAR depth is aligned in raw camera coordinates, then rendered in the iPhone display orientation.</p>
  <table>{rows}</table>

  <h2>Rendered Videos</h2>
  <div class="videos">
    <section>
      <h3>Projected LiDAR Depth Overlay</h3>
      <video controls src="outputs/depth_overlay.mp4"></video>
    </section>
    <section>
      <h3>Video Plus Sensor Fusion Panel</h3>
      <video controls src="outputs/fusion_panel.mp4"></video>
    </section>
  </div>

  <h2>Depth Projection Samples</h2>
  <img src="outputs/depth_projection_samples.png" alt="Depth projection samples">

  <h2>Sensor Fusion Dashboard</h2>
  <img src="outputs/sensor_fusion_dashboard.png" alt="Sensor fusion dashboard">

  <h2>World Point Cloud Map</h2>
  <img src="outputs/world_point_cloud_map.png" alt="World point cloud map">

  <p>Data exports: <a href="outputs/fused_timeline.csv">fused_timeline.csv</a>, <a href="outputs/world_point_cloud_sample.ply">world_point_cloud_sample.ply</a>.</p>
</main>
</body>
</html>
"""
    (output_dir.parent / "index.html").write_text(html)


def process_video_outputs(
    capture_dir: Path,
    output_dir: Path,
    video_info: VideoInfo,
    timeline: Timeline,
    depth_t: np.ndarray,
    depths: np.ndarray,
    vmin: float,
    vmax: float,
) -> list[tuple[int, Image.Image, Image.Image, Image.Image]]:
    video_path = capture_dir / "video.mp4"
    raw_w, raw_h = video_info.width, video_info.height
    display_size = (raw_h, raw_w)
    decoder = start_video_decoder(video_path, len(timeline.frame_t))
    assert decoder.stdout is not None

    overlay_encoder = start_raw_video_encoder(
        output_dir / "depth_overlay.mp4", display_size, video_info.fps_text, crf=22
    )
    fusion_encoder = start_raw_video_encoder(
        output_dir / "fusion_panel.mp4", (1280, 720), video_info.fps_text, crf=23
    )

    frame_bytes = raw_w * raw_h * 3
    sample_indices = set(
        int(i)
        for i in np.linspace(0, len(timeline.frame_t) - 1, num=min(4, len(timeline.frame_t)))
    )
    samples: list[tuple[int, Image.Image, Image.Image, Image.Image]] = []
    frame_i = 0
    try:
        while frame_i < len(timeline.frame_t):
            buf = decoder.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(raw_h, raw_w, 3)
            depth_low, _, _ = depth_for_time(float(timeline.frame_t[frame_i]), depth_t, depths)
            overlay_raw, dense = make_depth_overlay_raw(raw, depth_low, vmin, vmax)
            display_overlay = rotate_to_display(Image.fromarray(overlay_raw, mode="RGB"))
            display_overlay = annotate_overlay(
                display_overlay,
                float(timeline.rel_t[frame_i]),
                float(timeline.depth_median[frame_i]),
                float(timeline.depth_age_s[frame_i]),
                vmin,
                vmax,
            )
            write_frame(overlay_encoder, display_overlay)

            fusion_panel = make_fusion_panel(display_overlay, timeline, frame_i)
            write_frame(fusion_encoder, fusion_panel)

            if frame_i in sample_indices:
                display_video = rotate_to_display(Image.fromarray(raw, mode="RGB"))
                depth_norm = np.clip((dense - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
                depth_rgb = Image.fromarray((cm.turbo(depth_norm)[..., :3] * 255).astype(np.uint8), mode="RGB")
                depth_display = rotate_to_display(depth_rgb)
                samples.append((frame_i, display_video.copy(), depth_display.copy(), display_overlay.copy()))
            frame_i += 1
    finally:
        close_encoder(overlay_encoder, "depth overlay")
        close_encoder(fusion_encoder, "fusion panel")
        decoder.stdout.close()
        decoder.wait()
    if frame_i == 0:
        raise RuntimeError("No video frames decoded")
    if frame_i < len(timeline.frame_t):
        print(f"warning: decoded {frame_i} frames for {len(timeline.frame_t)} timeline entries", file=sys.stderr)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require_ffmpeg()

    intr = load_intrinsics(capture_dir / "intrinsics.json")
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    video_info = ffprobe_video(capture_dir / "video.mp4")
    poses = load_jsonl(capture_dir / "poses.jsonl")
    imu = load_jsonl(capture_dir / "imu.jsonl")
    depth_t, depths, depth_w, depth_h = load_depth(capture_dir)
    pose_t, pose_pos, pose_quat, imu_t, imu_accel, imu_gyro, imu_quat = rows_to_arrays(poses, imu)
    frame_times = frame_times_from_video(capture_dir / "video.mp4", video_info, pose_t[0])

    timeline = build_timeline(
        frame_times,
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
    valid_depth = depths[np.isfinite(depths) & (depths > 0)]
    vmin = float(np.percentile(valid_depth, 2))
    vmax = float(np.percentile(valid_depth, 98))

    print(f"capture: {capture_dir}")
    print(f"video: {video_info.width}x{video_info.height}, {video_info.nb_frames} frames, fps={video_info.fps_text}, rotation={video_info.rotation_deg:g}")
    print(f"depth: {depth_w}x{depth_h}, {len(depth_t)} frames, range={valid_depth.min():0.2f}-{valid_depth.max():0.2f}m")
    print(f"poses: {len(poses)} samples, imu: {len(imu)} samples")
    print(f"render depth range: {vmin:0.2f}-{vmax:0.2f}m")

    save_timeline_csv(output_dir / "fused_timeline.csv", timeline)
    samples = process_video_outputs(
        capture_dir,
        output_dir,
        video_info,
        timeline,
        depth_t,
        depths,
        vmin,
        vmax,
    )
    make_sample_grid(samples, output_dir / "depth_projection_samples.png")
    make_dashboard(output_dir / "sensor_fusion_dashboard.png", timeline, depth_t)
    make_world_point_cloud(
        output_dir / "world_point_cloud_map.png",
        output_dir / "world_point_cloud_sample.ply",
        intr,
        depth_t,
        depths,
        depth_w,
        depth_h,
        pose_t,
        pose_pos,
        pose_quat,
    )

    summary = {
        "recording": metadata.get("recordingId", "unknown"),
        "startedAt": metadata.get("startedAt", "unknown"),
        "duration": f"{video_info.duration_s:0.2f}s",
        "video": f"{video_info.width}x{video_info.height}, {video_info.nb_frames} frames, fps {video_info.fps_text}, display rotation {video_info.rotation_deg:g} deg",
        "depth": f"{depth_w}x{depth_h}, {len(depth_t)} frames, float32 metres",
        "pose": f"{len(poses)} camera 6DoF samples",
        "imu": f"{len(imu)} samples",
        "depth render range": f"{vmin:0.2f}m to {vmax:0.2f}m",
        "projection": "depth camera intrinsics are scaled from the 1920x1440 video intrinsics, then rendered into the video frame",
    }
    make_report(output_dir, summary)
    print(f"wrote report: {output_dir.parent / 'index.html'}")


if __name__ == "__main__":
    main()
