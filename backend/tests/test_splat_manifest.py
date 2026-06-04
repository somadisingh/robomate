import json
from pathlib import Path

from backend.splat.camera_path import build_camera_path
from backend.splat.manifest import (
    CameraPathArtifact,
    Intrinsics,
    SeedPointsArtifact,
    SplatArtifact,
    SplatManifest,
    TrainInfo,
)


def _sample_manifest() -> SplatManifest:
    return SplatManifest(
        splat=SplatArtifact(path="splat.spz", size_bytes=1234567, num_gaussians=250_000),
        camera_path=CameraPathArtifact(path="camera_path.json", frame_count=180, fps=30.0),
        seed_points=SeedPointsArtifact(path="seed_points.ply", point_count=42_000),
        train=TrainInfo(iterations=7000, gpu="A10G", duration_seconds=612.5),
        intrinsics=Intrinsics(
            fx=1462.0, fy=1462.0, cx=960.0, cy=540.0, width=1920, height=1080
        ),
    )


def test_manifest_to_dict_includes_version_and_components():
    manifest = _sample_manifest()
    payload = manifest.to_dict()
    assert payload["version"] == 1
    assert payload["splat"]["num_gaussians"] == 250_000
    assert payload["camera_path"]["frame_count"] == 180
    assert payload["seed_points"]["path"] == "seed_points.ply"
    assert payload["train"]["gpu"] == "A10G"
    assert payload["intrinsics"]["width"] == 1920


def test_manifest_db_summary_is_compact():
    manifest = _sample_manifest()
    assert manifest.db_summary() == {
        "num_gaussians": 250_000,
        "frame_count": 180,
        "fps": 30.0,
        "train_duration_seconds": 612.5,
        "iterations": 7000,
        "gpu": "A10G",
    }


def test_manifest_write_creates_file(tmp_path: Path):
    manifest = _sample_manifest()
    output = tmp_path / "manifest.json"
    manifest.write(output)
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded["version"] == 1
    assert loaded["splat"]["num_gaussians"] == 250_000


def test_build_camera_path_composes_dataparser_transform():
    # Simple test: dataparser identity scale=2 -> positions doubled.
    transforms = {
        "frames": [
            {
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "timestamp": 0.0,
                "video_frame_index": 0,
            },
            {
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 4.0],
                    [0.0, 0.0, 1.0, 6.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "timestamp": 1.0,
                "video_frame_index": 30,
            },
        ]
    }
    dataparser = {
        "transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "scale": 2.0,
    }
    payload = build_camera_path(transforms, dataparser)
    assert payload["count"] == 2
    assert payload["durationSeconds"] == 1.0
    assert payload["fps"] == 1.0
    assert payload["frames"][0]["position"] == [2.0, 4.0, 6.0]
    assert payload["frames"][1]["position"] == [4.0, 8.0, 12.0]
    assert payload["frames"][0]["videoFrameIndex"] == 0
    assert payload["frames"][1]["videoFrameIndex"] == 30


def test_build_camera_path_handles_single_frame():
    transforms = {
        "frames": [
            {
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "timestamp": 0.0,
                "video_frame_index": 0,
            }
        ]
    }
    dataparser = {
        "transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "scale": 1.0,
    }
    payload = build_camera_path(transforms, dataparser)
    assert payload["count"] == 1
    assert payload["durationSeconds"] == 0.0
    assert payload["fps"] == 0.0
