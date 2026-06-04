from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "outputs" / "nerfstudio" / "iphone-data-1"

APP_NAME = "copilot-hackathon-iphone-nerfstudio"
VOLUME_NAME = "copilot-hackathon-iphone-nerfstudio"
VOLUME_ROOT = Path("/workspace")
DATASETS_ROOT = VOLUME_ROOT / "datasets"
RUNS_ROOT = VOLUME_ROOT / "runs"
EXPORTS_ROOT = VOLUME_ROOT / "exports"
SPARK_CONVERTER_ROOT = Path("/opt/spark-converter")
SPARK_NODE_BINARY = SPARK_CONVERTER_ROOT / "node_modules" / "node" / "bin" / "node"
SPARK_CONVERTER_SCRIPT = SPARK_CONVERTER_ROOT / "convert-to-spz.mjs"

SPARK_CONVERTER_JS = r"""
import fs from "node:fs/promises";
import path from "node:path";
import { transcodeSpz } from "@sparkjsdev/spark";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error("usage: node convert-to-spz.mjs input.ply output.spz");
  process.exit(2);
}

const input = path.resolve(inputPath);
const output = path.resolve(outputPath);
const fileBytes = new Uint8Array(await fs.readFile(input));
const result = await transcodeSpz({
  inputs: [
    {
      fileBytes,
      pathOrUrl: input,
      transform: {
        translate: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: 1,
      },
    },
  ],
});

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, result.fileBytes);
console.log(`wrote ${output} (${result.fileBytes.length} bytes)`);
if (result.clippedCount) {
  console.log(`clipped ${result.clippedCount} splats`);
}
"""


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(
        "ghcr.io/nerfstudio-project/nerfstudio:latest",
        add_python="3.10",
    )
    .apt_install("nodejs", "npm")
    .run_commands(
        f"mkdir -p {shlex.quote(str(SPARK_CONVERTER_ROOT))}",
        f"cd {shlex.quote(str(SPARK_CONVERTER_ROOT))} && npm init -y && npm install node@20.11.1 @sparkjsdev/spark@2.1.0",
        "cat > "
        f"{shlex.quote(str(SPARK_CONVERTER_SCRIPT))} <<'EOF'\n"
        f"{SPARK_CONVERTER_JS}\n"
        "EOF",
    )
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
)
app = modal.App(APP_NAME, image=image)


def clean_dataset_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    if not cleaned:
        raise ValueError("dataset name is empty after sanitization")
    return cleaned


def run_command(cmd: list[str]) -> None:
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(["bash", "-lc", shlex.join(cmd)], check=True)


def identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
        for r in range(4)
    ]


def build_spark_camera_path(
    transforms_path: Path, dataparser_path: Path, output_path: Path
) -> dict:
    transforms = json.loads(transforms_path.read_text())
    dataparser = json.loads(dataparser_path.read_text())

    ns_transform = identity4()
    for r in range(3):
        for c in range(4):
            ns_transform[r][c] = float(dataparser["transform"][r][c])

    scale = float(dataparser["scale"])
    frames = []
    for frame in transforms["frames"]:
        matrix = multiply4(ns_transform, frame["transform_matrix"])
        for r in range(3):
            matrix[r][3] *= scale
        frames.append(
            {
                "videoFrameIndex": frame.get("video_frame_index"),
                "timestamp": frame["timestamp"],
                "position": [matrix[0][3], matrix[1][3], matrix[2][3]],
                "transformMatrix": matrix,
            }
        )

    positions = [frame["position"] for frame in frames]
    bounds = {
        "min": [min(position[axis] for position in positions) for axis in range(3)],
        "max": [max(position[axis] for position in positions) for axis in range(3)],
    }
    duration = frames[-1]["timestamp"] - frames[0]["timestamp"] if len(frames) > 1 else 0.0
    payload = {
        "source": {
            "transforms": transforms_path.name,
            "dataparserTransforms": dataparser_path.name,
        },
        "coordinateSpace": "nerfstudio_export",
        "count": len(frames),
        "durationSeconds": duration,
        "bounds": bounds,
        "frames": frames,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "path": str(output_path.relative_to(VOLUME_ROOT)),
        "pose_count": len(frames),
        "duration_seconds": duration,
        "bounds": bounds,
    }


def volume_path(path: Path) -> str:
    return str(path.relative_to(VOLUME_ROOT))


def download_volume_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        VOLUME_NAME,
        f"/{remote_path}",
        str(local_path),
    ]
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    gpu=["L4", "A10"],
    timeout=4 * 60 * 60,
)
def train_splatfacto_remote(
    dataset_name: str,
    max_num_iterations: int = 7000,
    method: str = "splatfacto",
    extra_args: str = "",
    skip_train: bool = False,
) -> dict:
    volume.reload()
    dataset_name = clean_dataset_name(dataset_name)
    dataset_dir = DATASETS_ROOT / dataset_name
    transforms_path = dataset_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"missing {transforms_path}")

    run_root = RUNS_ROOT / dataset_name
    export_dir = EXPORTS_ROOT / dataset_name / method
    run_root.mkdir(parents=True, exist_ok=True)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if skip_train:
        print(f"Skipping training; reusing latest config under {run_root}", flush=True)
    else:
        train_cmd = [
            "ns-train",
            method,
            "--data",
            str(dataset_dir),
            "--output-dir",
            str(run_root),
            "--max-num-iterations",
            str(max_num_iterations),
            "--viewer.quit-on-train-completion",
            "True",
        ]
        if extra_args:
            train_cmd.extend(shlex.split(extra_args))
        run_command(train_cmd)

    configs = sorted(
        run_root.glob("**/config.yml"), key=lambda path: path.stat().st_mtime
    )
    if not configs:
        raise RuntimeError(f"no Nerfstudio config.yml found under {run_root}")
    config_path = configs[-1]

    run_command(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_dir),
        ]
    )

    ply_path = export_dir / "splat.ply"
    spz_path = export_dir / "splat.spz"
    dataparser_path = config_path.parent / "dataparser_transforms.json"
    if not ply_path.exists():
        raise RuntimeError(f"missing expected Nerfstudio export {ply_path}")
    if not dataparser_path.exists():
        raise RuntimeError(f"missing expected Nerfstudio transform {dataparser_path}")

    run_command(
        [str(SPARK_NODE_BINARY), str(SPARK_CONVERTER_SCRIPT), str(ply_path), str(spz_path)]
    )
    shutil.copy2(config_path, export_dir / "config.yml")
    shutil.copy2(dataparser_path, export_dir / "dataparser_transforms.json")
    camera_path_info = build_spark_camera_path(
        transforms_path,
        export_dir / "dataparser_transforms.json",
        export_dir / "camera_path.json",
    )

    spark_manifest = {
        "dataset_name": dataset_name,
        "method": method,
        "spz": volume_path(spz_path),
        "camera_path": camera_path_info["path"],
        "dataparser_transforms": volume_path(export_dir / "dataparser_transforms.json"),
        "config": volume_path(export_dir / "config.yml"),
        "ply": volume_path(ply_path),
        "pose_count": camera_path_info["pose_count"],
        "duration_seconds": camera_path_info["duration_seconds"],
    }
    (export_dir / "spark_manifest.json").write_text(
        json.dumps(spark_manifest, indent=2) + "\n"
    )
    volume.commit()

    files = [
        volume_path(path)
        for path in sorted(export_dir.rglob("*"))
        if path.is_file()
    ]
    return {
        "dataset": volume_path(dataset_dir),
        "run_root": volume_path(run_root),
        "config": volume_path(config_path),
        "export_dir": volume_path(export_dir),
        "export_files": files,
        "spark": spark_manifest,
        "volume": VOLUME_NAME,
    }


@app.local_entrypoint()
def main(
    dataset_dir: str = str(DEFAULT_DATASET),
    dataset_name: str = "",
    max_num_iterations: int = 7000,
    method: str = "splatfacto",
    skip_upload: bool = False,
    upload_only: bool = False,
    download: bool = True,
    skip_train: bool = False,
    extra_args: str = "",
) -> None:
    local_dataset = Path(dataset_dir).expanduser().resolve()
    if not local_dataset.exists():
        raise FileNotFoundError(local_dataset)
    if not (local_dataset / "transforms.json").exists():
        raise FileNotFoundError(local_dataset / "transforms.json")

    name = clean_dataset_name(dataset_name or local_dataset.name)
    remote_dataset_path = f"/datasets/{name}"
    if not skip_upload:
        print(
            f"Uploading {local_dataset} to Modal volume "
            f"{VOLUME_NAME}:{remote_dataset_path}",
            flush=True,
        )
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_dataset), remote_dataset_path)
    else:
        print(f"Skipping upload; using {VOLUME_NAME}:{remote_dataset_path}", flush=True)

    if upload_only:
        print("Upload complete. Skipping remote training.", flush=True)
        return

    result = train_splatfacto_remote.remote(
        name,
        max_num_iterations=max_num_iterations,
        method=method,
        extra_args=extra_args,
        skip_train=skip_train,
    )
    print(json.dumps(result, indent=2), flush=True)
    if not download:
        print("Skipping local download.", flush=True)
        return

    local_export_dir = ROOT / "outputs" / "nerfstudio_modal_exports" / name
    spark_public = ROOT / "spark_viewer" / "public"
    downloads = {
        result["spark"]["spz"]: local_export_dir / "splat.spz",
        result["spark"]["camera_path"]: local_export_dir / "camera_path.json",
        result["spark"]["dataparser_transforms"]: local_export_dir
        / "dataparser_transforms.json",
        result["spark"]["config"]: local_export_dir / "config.yml",
        f"{result['export_dir']}/spark_manifest.json": local_export_dir
        / "spark_manifest.json",
    }
    for remote_file, local_file in downloads.items():
        download_volume_file(remote_file, local_file)

    splat_target = spark_public / "splats" / f"{name}.spz"
    camera_path_target = spark_public / "camera-paths" / f"{name}.json"
    splat_target.parent.mkdir(parents=True, exist_ok=True)
    camera_path_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_export_dir / "splat.spz", splat_target)
    shutil.copy2(local_export_dir / "camera_path.json", camera_path_target)

    print(f"Spark assets written for dataset {name}:", flush=True)
    print(f"  {splat_target}", flush=True)
    print(f"  {camera_path_target}", flush=True)
    print(f"Open http://127.0.0.1:5177/?dataset={name}", flush=True)
