"""Build the sparkjs camera-path JSON.

The path is in the same coordinate frame as the exported ``.spz`` (nerfstudio
applies a global ``transform`` + ``scale`` during training and bakes it into
the export). We compose the nerfstudio dataparser transform with each frame's
original transform_matrix so the result can be applied to a three.js camera
without further math.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _multiply4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
        for r in range(4)
    ]


def build_camera_path(transforms: dict[str, Any], dataparser: dict[str, Any]) -> dict[str, Any]:
    """Compose a sparkjs camera path payload from nerfstudio transforms.

    ``transforms`` is the contents of the dataset's ``transforms.json``.
    ``dataparser`` is the dataparser_transforms.json that nerfstudio writes
    alongside each training run.
    """
    ns_transform = _identity4()
    for r in range(3):
        for c in range(4):
            ns_transform[r][c] = float(dataparser["transform"][r][c])
    scale = float(dataparser["scale"])

    frames_out: list[dict[str, Any]] = []
    for frame in transforms["frames"]:
        matrix = _multiply4(ns_transform, frame["transform_matrix"])
        # Scale the translation column (the dataparser scale only applies to position).
        for r in range(3):
            matrix[r][3] *= scale
        frames_out.append(
            {
                "videoFrameIndex": frame.get("video_frame_index"),
                "timestamp": float(frame["timestamp"]),
                "position": [matrix[0][3], matrix[1][3], matrix[2][3]],
                "transformMatrix": matrix,
            }
        )

    duration = (
        float(frames_out[-1]["timestamp"] - frames_out[0]["timestamp"])
        if len(frames_out) > 1
        else 0.0
    )
    fps = float(len(frames_out) - 1) / duration if duration > 0 else 0.0
    positions = [frame["position"] for frame in frames_out]
    bounds = {
        "min": [min(p[axis] for p in positions) for axis in range(3)],
        "max": [max(p[axis] for p in positions) for axis in range(3)],
    } if positions else {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}

    return {
        "coordinateSpace": "nerfstudio_export",
        "count": len(frames_out),
        "durationSeconds": duration,
        "fps": fps,
        "bounds": bounds,
        "frames": frames_out,
    }


def write_camera_path(
    transforms_path: Path,
    dataparser_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    transforms = json.loads(Path(transforms_path).read_text())
    dataparser = json.loads(Path(dataparser_path).read_text())
    payload = build_camera_path(transforms, dataparser)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
