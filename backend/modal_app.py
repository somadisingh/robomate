from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import modal

from backend.artifacts import (
    analysis_artifact_paths,
    detected_object_summary,
    gaussian_splat_dir,
    normalize_sam_prompts,
)
from backend.contracts import ANALYSIS_KINDS, AnalysisKind, AnalysisRequest
from backend.modal_inference.hand_landmarks import ensure_hand_model, infer_hands
from backend.modal_inference.media import (
    bounded_max_frames,
    is_video_suffix,
    output_json as write_output_json,
    write_media_bytes,
)
from backend.modal_inference.ultralytics_results import result_to_record
from backend.orchestrator import (
    fetch_context,
    finalize_multi_artifact_job,
    gaussian_splat_preflight,
    mark_job,
    prune_resource_intensive_jobs,
    run_gemini,
    run_recording_index_step,
    run_remote_analyzer,
    run_temporal_annotation_step,
    update_final_status,
    utc_now,
)
from backend.supabase_api import SupabaseApi, SupabaseConfig

try:
    from fastapi import Request
except ImportError:
    Request = Any


APP_NAME = "copilot-hackathon-backend-analysis"
MODEL_VOLUME_NAME = "copilot-hackathon-modal-inference-models"
MODEL_ROOT = "/models"
SECRET_NAME = "copilot-hackathon-backend-secrets"

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
BACKEND_PACKAGE_DIR = THIS_DIR / "backend"
TAS_SOURCE_DIR = (
    REPO_ROOT
    / "playground"
    / "temporal_action_segmentation"
    / "temporal_action_segmentation"
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
backend_secret = modal.Secret.from_name(SECRET_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgles2", "libglib2.0-0", "libgl1")
    .pip_install(
        "httpx>=0.28.0",
        "numpy>=2.2.0",
        "opencv-python-headless>=4.13.0.92",
        "pydantic>=2.7.0",
    )
)

yolo_image = base_image.pip_install("ultralytics>=8.4.53").add_local_dir(
    BACKEND_PACKAGE_DIR,
    remote_path="/root/backend",
    ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
)

sam_image = base_image.pip_install(
    "ultralytics>=8.4.53",
    "git+https://github.com/ultralytics/CLIP.git",
    "timm>=1.0.0",
).add_local_dir(
    BACKEND_PACKAGE_DIR,
    remote_path="/root/backend",
    ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
)

mediapipe_runtime_image = base_image.pip_install("mediapipe>=0.10.35")

mediapipe_image = mediapipe_runtime_image.add_local_dir(
    BACKEND_PACKAGE_DIR,
    remote_path="/root/backend",
    ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
)

tas_runtime_image = mediapipe_runtime_image.pip_install("google-genai>=1.0.0")

tas_image = (
    tas_runtime_image.add_local_dir(
        BACKEND_PACKAGE_DIR,
        remote_path="/root/backend",
        ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
    )
    .add_local_dir(
        TAS_SOURCE_DIR,
        remote_path="/root/temporal_action_segmentation",
        ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
    )
)

SPARK_CONVERTER_ROOT = "/opt/spark-converter"
SPARK_CONVERTER_SCRIPT = f"{SPARK_CONVERTER_ROOT}/convert-to-spz.mjs"

nerfstudio_image = (
    modal.Image.from_registry(
        "ghcr.io/nerfstudio-project/nerfstudio:latest",
        add_python="3.10",
    )
    .apt_install("nodejs", "npm", "ffmpeg")
    .pip_install(
        "httpx>=0.28.0",
        "numpy>=2.2.0",
        "pillow>=10.0.0",
        "pydantic>=2.7.0",
    )
    .run_commands(
        f"mkdir -p {SPARK_CONVERTER_ROOT}",
        f"cd {SPARK_CONVERTER_ROOT} && npm init -y && "
        "npm install node@20.11.1 @sparkjsdev/spark@2.1.0",
    )
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
    .add_local_dir(
        BACKEND_PACKAGE_DIR,
        remote_path="/root/backend",
        ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
    )
)

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx>=0.28.0",
        "google-genai>=1.0.0",
        "pinecone>=5.0.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.7.0",
        "python-dotenv>=1.0.1",
    )
    .add_local_dir(
        BACKEND_PACKAGE_DIR,
        remote_path="/root/backend",
        ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
    )
)

REMOTE_ANALYSIS_KINDS: tuple[AnalysisKind, ...] = (
    "yolo_objects",
    "mediapipe_hands",
    "sam_segments",
    "temporal_actions",
)

# The splat kind requires a depth-aware preflight, so it isn't part of
# REMOTE_ANALYSIS_KINDS (which all run unconditionally on every submission).
GAUSSIAN_SPLAT_KIND: AnalysisKind = "gaussian_splat"


def resource_intensive_analysis_enabled() -> bool:
    return os.environ.get("COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cuda_device() -> int | str:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _upload_output(payload: object, output_path: str) -> None:
    if output_path:
        write_output_json(Path(output_path).expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


class _GeminiTemporalLabel:
    def __init__(
        self,
        *,
        meaningful_manipulation: bool,
        caption: str,
        object_name: str | None,
        confidence: float,
        reason: str,
    ) -> None:
        self.meaningful_manipulation = meaningful_manipulation
        self.caption = caption
        self.object_name = object_name
        self.confidence = confidence
        self.reason = reason

    def as_record(self) -> dict[str, object]:
        return {
            "meaningful_manipulation": self.meaningful_manipulation,
            "caption": self.caption,
            "object": self.object_name,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class GeminiTemporalLabeler:
    def __init__(self, *, model: str, cache_dir: Path) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for temporal action labelling.")

        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def label(self, segment: object, contact_sheet_path: Path) -> _GeminiTemporalLabel:
        cache_path = self._cache_path(segment, contact_sheet_path)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return self._normalize(cached["label"])

        from google.genai import types

        image_bytes = contact_sheet_path.read_bytes()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                self._prompt(segment),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty temporal action label.")

        payload = _extract_json_object(text)
        label = self._normalize(payload)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "prompt_version": "tas-gemini-v1",
                    "label": label.as_record(),
                    "raw_response": text,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return label

    def _cache_path(self, segment: object, contact_sheet_path: Path) -> Path:
        digest = hashlib.sha256()
        digest.update(b"tas-gemini-v1")
        digest.update(self.model.encode("utf-8"))
        digest.update(str(getattr(segment, "hand", "")).encode("utf-8"))
        digest.update(str(getattr(segment, "start_frame", "")).encode("ascii"))
        digest.update(str(getattr(segment, "end_frame", "")).encode("ascii"))
        digest.update(contact_sheet_path.read_bytes())
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _prompt(self, segment: object) -> str:
        hand = getattr(segment, "hand", "tracked")
        start_sec = float(getattr(segment, "start_sec", 0.0))
        end_sec = float(getattr(segment, "end_sec", 0.0))
        return f"""You are annotating egocentric hand-manipulation video clips.
The contact sheet shows sampled frames from one short clip, with a colored trajectory marking the {hand} hand.
Describe only the action performed by that hand.
Use an imperative robot-instruction style, for example "Pick up the mug", "Open the drawer", "Wipe the counter".

Return strict JSON only:
{{
  "meaningful_manipulation": true,
  "caption": "short imperative caption or N/A",
  "object": "object name or null",
  "confidence": 0.0,
  "reason": "short reason"
}}

Rules:
- If the hand is idle, gesturing, occluded, or not manipulating an object, return meaningful_manipulation false and caption "N/A".
- Prefer short atomic actions.
- Do not describe camera motion.
- Do not invent objects that are not visible.
- The clip spans {start_sec:.2f}s to {end_sec:.2f}s in the source video."""

    def _normalize(self, payload: dict[str, object]) -> _GeminiTemporalLabel:
        meaningful = bool(payload.get("meaningful_manipulation", False))
        caption = str(payload.get("caption") or "N/A").strip()
        if not meaningful:
            caption = "N/A"

        raw_confidence = payload.get("confidence", 0.0) or 0.0
        confidence = min(1.0, max(0.0, float(raw_confidence)))
        raw_object = payload.get("object")
        object_name = None if raw_object in ("", "null", None) else str(raw_object)

        return _GeminiTemporalLabel(
            meaningful_manipulation=meaningful,
            caption=caption,
            object_name=object_name,
            confidence=confidence,
            reason=str(payload.get("reason") or ""),
        )


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in Gemini response: {text[:200]}")
    return json.loads(stripped[start : end + 1])


@app.cls(
    image=yolo_image,
    gpu=["L4", "T4", "A10"],
    volumes={MODEL_ROOT: model_volume},
    timeout=15 * 60,
    scaledown_window=60,
    max_containers=2,
)
class Yolo26:
    @modal.enter()
    def setup(self) -> None:
        self._models = {}

    def _model(self, task: str):
        from ultralytics import YOLO

        filenames = {
            "detect": "yolo26n.pt",
            "instance": "yolo26n-seg.pt",
        }
        if task not in filenames:
            raise ValueError("YOLO26 task must be 'detect' or 'instance'.")

        filename = filenames[task]
        if filename in self._models:
            return self._models[filename], filename

        model_dir = Path(MODEL_ROOT) / "yolo"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            old_cwd = Path.cwd()
            os.chdir(model_dir)
            try:
                model = YOLO(filename)
                if model_path.exists():
                    model_volume.commit()
                    model = YOLO(str(model_path))
            finally:
                os.chdir(old_cwd)
        self._models[filename] = model
        return model, filename

    @modal.method()
    def predict(
        self,
        media: bytes,
        *,
        suffix: str = ".mp4",
        task: Literal["detect", "instance"] = "detect",
        conf: float = 0.25,
        imgsz: int = 640,
        vid_stride: int = 1,
        max_frames: int | None = 300,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = write_media_bytes(media, Path(tmp), suffix)
            model, filename = self._model(task)
            results = model.predict(
                source=str(input_path),
                stream=True,
                conf=conf,
                imgsz=imgsz,
                vid_stride=max(1, vid_stride),
                device=_cuda_device(),
                verbose=False,
            )

            frames = []
            limit = bounded_max_frames(max_frames)
            for frame_index, result in enumerate(results):
                if limit is not None and frame_index >= limit:
                    break
                frames.append(
                    result_to_record(
                        result,
                        frame_index,
                        include_masks=task == "instance",
                    )
                )

        return {
            "engine": "ultralytics-yolo26",
            "task": task,
            "model": filename,
            "frame_count": len(frames),
            "frames": frames,
            "settings": {"conf": conf, "imgsz": imgsz, "vid_stride": max(1, vid_stride)},
        }


@app.cls(
    image=sam_image,
    gpu=["H100", "A100-80GB", "L40S"],
    volumes={MODEL_ROOT: model_volume},
    timeout=20 * 60,
    scaledown_window=60,
    max_containers=1,
)
class SAM31Segmenter:
    @modal.enter()
    def setup(self) -> None:
        self._predictors = {}

    def _checkpoint_path(self, model_path: str | None) -> Path:
        path = Path(model_path or os.environ.get("SAM31_MODEL_PATH", f"{MODEL_ROOT}/sam/sam3.pt"))
        if not path.exists():
            raise FileNotFoundError(
                "SAM 3.1 checkpoint not found. Upload it to the Modal Volume, for example: "
                f"modal volume put {MODEL_VOLUME_NAME} /path/to/sam3.pt /sam/sam3.pt"
            )
        return path

    def _predictor(self, *, video: bool, checkpoint: Path, conf: float, imgsz: int):
        key = (video, str(checkpoint), conf, imgsz)
        if key in self._predictors:
            return self._predictors[key]

        overrides = {
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(checkpoint),
            "imgsz": imgsz,
            "half": True,
            "verbose": False,
        }
        if video:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor

            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            self._predictors[key] = predictor
            return predictor

        from ultralytics.models.sam import SAM3SemanticPredictor

        predictor = SAM3SemanticPredictor(overrides=overrides)
        self._predictors[key] = predictor
        return predictor

    @modal.method()
    def segment(
        self,
        media: bytes,
        *,
        suffix: str = ".mp4",
        text_prompts: list[str] | None = None,
        conf: float = 0.25,
        imgsz: int = 512,
        max_frames: int | None = 300,
        model_path: str | None = None,
    ) -> dict[str, object]:
        prompts = [prompt.strip() for prompt in (text_prompts or []) if prompt.strip()]
        if not prompts:
            raise ValueError("SAM 3.1 concept segmentation requires at least one text prompt.")

        checkpoint = self._checkpoint_path(model_path)
        with tempfile.TemporaryDirectory() as tmp:
            input_path = write_media_bytes(media, Path(tmp), suffix)
            video = is_video_suffix(suffix)
            predictor_t0 = time.perf_counter()
            predictor = self._predictor(video=video, checkpoint=checkpoint, conf=conf, imgsz=imgsz)
            predictor_load_s = time.perf_counter() - predictor_t0

            inference_t0 = time.perf_counter()
            results = predictor(source=str(input_path), text=prompts, stream=True)

            frames = []
            limit = bounded_max_frames(max_frames)
            for frame_index, result in enumerate(results):
                if limit is not None and frame_index >= limit:
                    break
                frames.append(
                    result_to_record(
                        result,
                        frame_index,
                        include_masks=True,
                    )
                )
            inference_s = time.perf_counter() - inference_t0

        timing = {
            "predictor_load_s": round(predictor_load_s, 3),
            "inference_s": round(inference_s, 3),
            "per_frame_ms": round(inference_s * 1000 / max(1, len(frames)), 2),
        }
        print(f"[SAM31] frames={len(frames)} imgsz={imgsz} timing={timing}")

        return {
            "engine": "sam-3.1-ultralytics",
            "task": "instance",
            "model": str(checkpoint),
            "text_prompts": prompts,
            "frame_count": len(frames),
            "frames": frames,
            "settings": {"conf": conf, "imgsz": imgsz},
            "timing": timing,
        }


@app.cls(
    image=mediapipe_image,
    volumes={MODEL_ROOT: model_volume},
    cpu=2.0,
    memory=4096,
    timeout=15 * 60,
    scaledown_window=30,
    max_containers=2,
)
class MediaPipeHands:
    @modal.enter()
    def setup(self) -> None:
        model_path = Path(MODEL_ROOT) / "mediapipe" / "hand_landmarker.task"
        already_cached = model_path.exists()
        self.model_path = ensure_hand_model(model_path)
        if not already_cached:
            model_volume.commit()

    @modal.method()
    def landmarks(
        self,
        media: bytes,
        *,
        suffix: str = ".mp4",
        target_fps: float = 10.0,
        max_frames: int | None = 300,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = write_media_bytes(media, Path(tmp), suffix)
            payload = infer_hands(
                input_path,
                self.model_path,
                is_video=is_video_suffix(suffix),
                target_fps=target_fps,
                max_frames=bounded_max_frames(max_frames),
                max_hands=max_hands,
                detection_confidence=detection_confidence,
                presence_confidence=presence_confidence,
                tracking_confidence=tracking_confidence,
            )

        return {
            "engine": "mediapipe-hand-landmarker",
            "model": str(self.model_path),
            "settings": {
                "target_fps": target_fps,
                "max_hands": max_hands,
                "detection_confidence": detection_confidence,
                "presence_confidence": presence_confidence,
                "tracking_confidence": tracking_confidence,
            },
            **payload,
        }


@app.cls(
    image=tas_image,
    secrets=[backend_secret],
    volumes={MODEL_ROOT: model_volume},
    cpu=2.0,
    memory=6144,
    timeout=30 * 60,
    scaledown_window=30,
    max_containers=1,
)
class TemporalActionSegmenter:
    @modal.enter()
    def setup(self) -> None:
        from temporal_action_segmentation.hand_tracking import ensure_model

        model_path = Path(MODEL_ROOT) / "mediapipe" / "hand_landmarker.task"
        already_cached = model_path.exists()
        self.model_path = ensure_model(model_path)
        if not already_cached:
            model_volume.commit()

    @modal.method()
    def segment(
        self,
        video: bytes,
        *,
        suffix: str = ".mp4",
        target_fps: float = 10.0,
        min_seg_s: float = 0.6,
        max_seg_s: float = 6.0,
        min_visible_ratio: float = 0.6,
        min_motion: float = 0.01,
        max_segments: int | None = 200,
    ) -> dict[str, object]:
        if not is_video_suffix(suffix):
            raise ValueError("Temporal action segmentation expects a video input.")

        from temporal_action_segmentation.pipeline import PipelineConfig, process_video

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = write_media_bytes(video, tmp_path, suffix)
            output_dir = tmp_path / "outputs"
            config = PipelineConfig(
                output_dir=output_dir,
                model_path=self.model_path,
                target_fps=target_fps,
                min_seg_s=min_seg_s,
                max_seg_s=max_seg_s,
                min_visible_ratio=min_visible_ratio,
                min_motion=min_motion,
                render_contact_sheets=True,
                write_review=False,
                labeler="gemini",
                openai_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                max_segments=max_segments,
            )
            labeler = GeminiTemporalLabeler(
                model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                cache_dir=output_dir / "cache",
            )
            records = process_video(input_path, config, labeler)
            records.sort(key=lambda item: (item["video_id"], item["hand"], item["start_sec"]))

        return {
            "engine": "copilot-hackathon-temporal-action-segmentation",
            "model": str(self.model_path),
            "labeler": {
                "engine": "gemini",
                "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
            },
            "segment_count": len(records),
            "segments": records,
            "settings": {
                "target_fps": target_fps,
                "min_seg_s": min_seg_s,
                "max_seg_s": max_seg_s,
                "min_visible_ratio": min_visible_ratio,
                "min_motion": min_motion,
                "max_segments": max_segments,
            },
        }


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
  inputs: [{
    fileBytes,
    pathOrUrl: input,
    transform: { translate: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
  }],
});

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, result.fileBytes);
console.log(`wrote ${output} (${result.fileBytes.length} bytes)`);
if (result.clippedCount) {
  console.log(`clipped ${result.clippedCount} splats`);
}
"""


@app.cls(
    image=nerfstudio_image,
    secrets=[backend_secret],
    gpu="A10G",
    timeout=40 * 60,
    scaledown_window=60,
    max_containers=2,
)
class SplatfactoTrainer:
    @modal.enter()
    def setup(self) -> None:
        from pathlib import Path as _P
        script_path = _P(SPARK_CONVERTER_SCRIPT)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        if not script_path.exists():
            script_path.write_text(SPARK_CONVERTER_JS)

    @modal.method()
    def train(
        self,
        recording_id: str,
        storage_path: str,
        *,
        max_num_iterations: int = 7000,
        method: str = "splatfacto",
        bucket: str = "recordings",
        update_db: bool = True,
    ) -> dict[str, Any]:
        import shlex
        import shutil
        import time as _time

        from backend.splat.camera_path import write_camera_path
        from backend.splat.convert import (
            SPARK_CONVERTER_SCRIPT as _SCRIPT,
            ply_to_spz,
        )
        from backend.splat.dataset import DatasetConfig, build_dataset
        from backend.splat.manifest import (
            CameraPathArtifact,
            Intrinsics,
            SeedPointsArtifact,
            SplatArtifact,
            SplatManifest,
            TrainInfo,
        )

        # Ensure the spark converter script is materialised (idempotent).
        _SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        if not _SCRIPT.exists():
            _SCRIPT.write_text(SPARK_CONVERTER_JS)

        api = SupabaseApi(SupabaseConfig.from_service_role_env())

        def _safe_db(label: str, fn) -> None:
            if not update_db:
                return
            try:
                fn()
            except Exception as exc:  # best-effort observability
                print(f"[splat] {label} DB write failed: {exc}", flush=True)

        # Best-effort: mark the job 'running' inside the container so the
        # studio sees status without depending on the local entrypoint creds.
        _safe_db(
            "upsert running",
            lambda: api.upsert_rows(
                "recording_analysis_jobs",
                [
                    {
                        "recording_id": recording_id,
                        "kind": "gaussian_splat",
                        "status": "running",
                        "started_at": utc_now(),
                        "finished_at": None,
                        "error": None,
                        "artifact_path": analysis_artifact_paths(
                            recording_id
                        )["gaussian_splat"],
                    }
                ],
                on_conflict="recording_id,kind",
            ),
        )

        try:
            storage_prefix = storage_path.strip().strip("/") or recording_id
            workspace = Path(tempfile.mkdtemp(prefix="splat-"))
            try:
                # 1) Pull required streams + recording row.
                video_bytes = api.download_bytes(bucket, f"{storage_prefix}/video.mp4")
                depth_bytes = api.download_bytes(bucket, f"{storage_prefix}/depth.bin")
                poses_text = api.download_bytes(
                    bucket, f"{storage_prefix}/poses.jsonl"
                ).decode("utf-8")
                intrinsics_payload = json.loads(
                    api.download_bytes(
                        bucket, f"{storage_prefix}/intrinsics.json"
                    ).decode("utf-8")
                )

                row = api.select_one(
                    "recordings",
                    f"id=eq.{recording_id}"
                    "&select=depth_width,depth_height,depth_frame_count",
                )
                depth_width = int(row["depth_width"])
                depth_height = int(row["depth_height"])

                video_path = workspace / "video.mp4"
                depth_path = workspace / "depth.bin"
                video_path.write_bytes(video_bytes)
                depth_path.write_bytes(depth_bytes)

                dataset_dir = workspace / "dataset"
                run_root = workspace / "runs"
                export_dir = workspace / "exports"
                run_root.mkdir(parents=True, exist_ok=True)
                export_dir.mkdir(parents=True, exist_ok=True)

                # 2) Build the nerfstudio dataset (frames + transforms + sparse PC).
                dataset_summary = build_dataset(
                    output_dir=dataset_dir,
                    video_path=video_path,
                    depth_path=depth_path,
                    poses_text=poses_text,
                    intrinsics=intrinsics_payload,
                    depth_width=depth_width,
                    depth_height=depth_height,
                    config=DatasetConfig(),
                )

                # 3) ns-train splatfacto.
                train_started = _time.monotonic()
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
                print("$", shlex.join(train_cmd), flush=True)
                subprocess.run(
                    ["bash", "-lc", shlex.join(train_cmd)], check=True
                )

                configs = sorted(
                    run_root.glob("**/config.yml"),
                    key=lambda path: path.stat().st_mtime,
                )
                if not configs:
                    raise RuntimeError("no nerfstudio config.yml emitted by ns-train")
                config_path = configs[-1]

                # 4) ns-export gaussian-splat → PLY.
                export_cmd = [
                    "ns-export",
                    "gaussian-splat",
                    "--load-config",
                    str(config_path),
                    "--output-dir",
                    str(export_dir),
                ]
                print("$", shlex.join(export_cmd), flush=True)
                subprocess.run(
                    ["bash", "-lc", shlex.join(export_cmd)], check=True
                )

                ply_path = export_dir / "splat.ply"
                spz_path = export_dir / "splat.spz"
                dataparser_path = config_path.parent / "dataparser_transforms.json"
                if not ply_path.exists():
                    raise RuntimeError(f"ns-export missing {ply_path}")
                if not dataparser_path.exists():
                    raise RuntimeError(f"ns-export missing {dataparser_path}")

                # 5) PLY → SPZ.
                ply_to_spz(ply_path, spz_path)

                # 6) camera_path.json in the splat's coordinate frame.
                camera_path_payload = write_camera_path(
                    transforms_path=dataset_dir / "transforms.json",
                    dataparser_path=dataparser_path,
                    output_path=export_dir / "camera_path.json",
                )

                # 7) train_config.json (snapshot for repro).
                train_config_dest = export_dir / "train_config.json"
                train_config_dest.write_text(
                    json.dumps(
                        {
                            "method": method,
                            "iterations": max_num_iterations,
                            "config_yml": config_path.read_text(),
                        },
                        indent=2,
                    )
                )

                train_duration = _time.monotonic() - train_started
                seed_ply_path = dataset_dir / "sparse_pc.ply"
                seed_ply_count = dataset_summary.point_count

                num_gaussians = _count_ply_vertices(ply_path)

                manifest = SplatManifest(
                    splat=SplatArtifact(
                        path="splat.spz",
                        size_bytes=spz_path.stat().st_size,
                        num_gaussians=num_gaussians,
                    ),
                    camera_path=CameraPathArtifact(
                        path="camera_path.json",
                        frame_count=int(camera_path_payload["count"]),
                        fps=float(camera_path_payload.get("fps") or dataset_summary.fps),
                    ),
                    seed_points=SeedPointsArtifact(
                        path="seed_points.ply",
                        point_count=seed_ply_count,
                    ),
                    train=TrainInfo(
                        iterations=max_num_iterations,
                        gpu="A10G",
                        duration_seconds=float(train_duration),
                        method=method,
                    ),
                    intrinsics=Intrinsics(
                        fx=float(intrinsics_payload["fx"]),
                        fy=float(intrinsics_payload["fy"]),
                        cx=float(intrinsics_payload["cx"]),
                        cy=float(intrinsics_payload["cy"]),
                        width=int(intrinsics_payload["width"]),
                        height=int(intrinsics_payload["height"]),
                    ),
                )
                manifest_path = export_dir / "manifest.json"
                manifest.write(manifest_path)

                # 8) Upload artifacts to Supabase under analysis/gaussian_splat/.
                splat_prefix = gaussian_splat_dir(recording_id)
                api.upload_bytes(
                    bucket,
                    f"{splat_prefix}/splat.spz",
                    spz_path.read_bytes(),
                    "application/octet-stream",
                )
                api.upload_bytes(
                    bucket,
                    f"{splat_prefix}/camera_path.json",
                    (export_dir / "camera_path.json").read_bytes(),
                    "application/json",
                )
                api.upload_bytes(
                    bucket,
                    f"{splat_prefix}/seed_points.ply",
                    seed_ply_path.read_bytes(),
                    "application/octet-stream",
                )
                api.upload_bytes(
                    bucket,
                    f"{splat_prefix}/train_config.json",
                    train_config_dest.read_bytes(),
                    "application/json",
                )
                api.upload_bytes(
                    bucket,
                    f"{splat_prefix}/manifest.json",
                    manifest_path.read_bytes(),
                    "application/json",
                )

                manifest_storage_path = f"{splat_prefix}/manifest.json"
                summary = manifest.db_summary()
                _safe_db(
                    "mark succeeded",
                    lambda: mark_job(
                        api,
                        recording_id,
                        "gaussian_splat",
                        "succeeded",
                        artifact_path=manifest_storage_path,
                        summary=summary,
                        error=None,
                        finished_at=utc_now(),
                    ),
                )
                return {
                    "manifest_path": manifest_storage_path,
                    "summary": summary,
                }
            finally:
                shutil.rmtree(workspace, ignore_errors=True)
        except Exception as exc:
            _safe_db(
                "mark failed",
                lambda: mark_job(
                    api,
                    recording_id,
                    "gaussian_splat",
                    "failed",
                    error=_error_message(exc),
                    finished_at=utc_now(),
                ),
            )
            raise
        finally:
            api.close()


def _count_ply_vertices(ply_path: Path) -> int:
    """Read a PLY header and return ``element vertex N`` count (binary or ascii)."""
    with ply_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                text = line.decode("ascii", errors="replace").strip()
            except Exception:
                continue
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                break
    return 0


@app.function(
    image=orchestrator_image,
    secrets=[backend_secret],
    timeout=45 * 60,
    scaledown_window=60,
    max_containers=4,
)
def process_recording(payload: dict) -> dict[str, object]:
    request = AnalysisRequest.model_validate(payload)
    api = SupabaseApi(SupabaseConfig.from_service_role_env())
    failed = False

    try:
        try:
            task, recording, video_bytes = fetch_context(api, request)
        except Exception as exc:
            message = _error_message(exc)
            for kind in ANALYSIS_KINDS:
                mark_job(
                    api,
                    request.recording_id,
                    kind,
                    "failed",
                    error=message,
                    finished_at=utc_now(),
                )
            api.patch_rows(
                "recordings",
                f"id=eq.{request.recording_id}",
                {"status": "analysis_failed", "is_scoring": False},
            )
            raise

        api.patch_rows("recordings", f"id=eq.{request.recording_id}", {"status": "analyzing"})
        prompts = normalize_sam_prompts(task.objects)

        run_resource_intensive = resource_intensive_analysis_enabled()
        if not run_resource_intensive:
            prune_resource_intensive_jobs(api, request.recording_id)

        splat_eligible = run_resource_intensive and gaussian_splat_preflight(recording)
        if splat_eligible:
            # The submit-recording Edge Function doesn't pre-create gaussian_splat
            # rows (preflight is depth-dependent), so we insert one here before
            # marking it running.
            api.upsert_rows(
                "recording_analysis_jobs",
                [
                    {
                        "recording_id": request.recording_id,
                        "kind": GAUSSIAN_SPLAT_KIND,
                        "status": "pending",
                        "artifact_path": analysis_artifact_paths(
                            request.recording_id
                        )[GAUSSIAN_SPLAT_KIND],
                    }
                ],
                on_conflict="recording_id,kind",
            )

        gemini_score: float | None = None
        # Parsed analyzer payloads kept in memory so the Pinecone recording
        # indexer (Feature 1) can aggregate them at the end of the pipeline.
        analysis_outputs: dict[str, Any] = {}
        futures = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures[executor.submit(run_gemini, api, request, task, video_bytes)] = "gemini_eval"

            if run_resource_intensive:
                for kind in REMOTE_ANALYSIS_KINDS:
                    mark_job(
                        api,
                        request.recording_id,
                        kind,
                        "running",
                        error=None,
                        started_at=utc_now(),
                        finished_at=None,
                    )

                futures[
                    executor.submit(
                        lambda: Yolo26().predict.remote(
                            video_bytes,
                            suffix=".mp4",
                            task="detect",
                            max_frames=None,
                        )
                    )
                ] = "yolo_objects"
                futures[
                    executor.submit(
                        lambda: MediaPipeHands().landmarks.remote(
                            video_bytes,
                            suffix=".mp4",
                            target_fps=10.0,
                            max_frames=None,
                        )
                    )
                ] = "mediapipe_hands"
                futures[
                    executor.submit(
                        lambda: SAM31Segmenter().segment.remote(
                            video_bytes,
                            suffix=".mp4",
                            text_prompts=prompts,
                            max_frames=None,
                        )
                    )
                ] = "sam_segments"
                futures[
                    executor.submit(
                        lambda: TemporalActionSegmenter().segment.remote(
                            video_bytes,
                            suffix=".mp4",
                            max_segments=200,
                        )
                    )
                ] = "temporal_actions"

            if splat_eligible:
                mark_job(
                    api,
                    request.recording_id,
                    GAUSSIAN_SPLAT_KIND,
                    "running",
                    error=None,
                    started_at=utc_now(),
                    finished_at=None,
                )
                futures[
                    executor.submit(
                        lambda: SplatfactoTrainer().train.remote(
                            request.recording_id,
                            request.storage_path,
                        )
                    )
                ] = GAUSSIAN_SPLAT_KIND

            for future in as_completed(futures):
                kind = futures[future]
                try:
                    artifact_payload = future.result()
                except Exception as exc:
                    failed = True
                    mark_job(
                        api,
                        request.recording_id,
                        kind,
                        "failed",
                        error=_error_message(exc),
                        finished_at=utc_now(),
                    )
                    if kind == "gemini_eval":
                        api.patch_rows(
                            "recordings",
                            f"id=eq.{request.recording_id}",
                            {"is_scoring": False},
                        )
                    continue

                if kind == "gemini_eval":
                    if isinstance(artifact_payload, dict):
                        gemini_score = artifact_payload.get("score")
                        analysis_outputs["gemini_eval"] = artifact_payload
                    continue

                if kind == GAUSSIAN_SPLAT_KIND:
                    try:
                        finalize_multi_artifact_job(
                            api,
                            request,
                            GAUSSIAN_SPLAT_KIND,
                            artifact_path=artifact_payload["manifest_path"],
                            db_summary=artifact_payload.get("summary"),
                        )
                    except Exception as exc:
                        failed = True
                        mark_job(
                            api,
                            request.recording_id,
                            kind,
                            "failed",
                            error=_error_message(exc),
                            finished_at=utc_now(),
                        )
                    continue

                try:
                    if kind in ("yolo_objects", "temporal_actions") and isinstance(artifact_payload, dict):
                        analysis_outputs[kind] = artifact_payload
                    summary = detected_object_summary(artifact_payload) if kind == "yolo_objects" else None
                    run_remote_analyzer(api, request, kind, artifact_payload, summary)
                    if kind == "yolo_objects":
                        api.patch_rows(
                            "recordings",
                            f"id=eq.{request.recording_id}",
                            {"detected_objects": summary},
                        )
                except Exception as exc:
                    failed = True
                    mark_job(
                        api,
                        request.recording_id,
                        kind,
                        "failed",
                        error=_error_message(exc),
                        finished_at=utc_now(),
                    )
                    continue

        # Gemini temporal annotations. Runs after the analyzers above so the
        # Gemini quality score is available. Fully isolated: it records its own
        # job status and never raises. Returns the parsed annotations (or None).
        temporal_annotations = run_temporal_annotation_step(
            api, request, task, video_bytes, gemini_score
        )

        # Feature 1: aggregate all outputs into one Pinecone "recordings" vector.
        # Post-processing only (not tracked in recording_analysis_jobs).
        run_recording_index_step(
            api,
            request,
            task,
            recording,
            analysis_outputs,
            temporal_annotations,
            gemini_score,
        )

        final_status = update_final_status(api, request.recording_id)
        return {
            "ok": not failed,
            "recording_id": request.recording_id,
            "status": final_status,
        }
    finally:
        api.close()


@app.function(
    image=orchestrator_image,
    secrets=[backend_secret],
    timeout=60,
    scaledown_window=30,
)
@modal.fastapi_endpoint(method="POST")
def submit_analysis(payload: dict[str, Any], request: Request):
    from fastapi import HTTPException

    expected = os.environ.get("MODAL_ANALYSIS_SECRET")
    received = request.headers.get("X-Copilot-Hackathon-Modal-Secret")
    if not expected or received != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    parsed = AnalysisRequest.model_validate(payload)
    call = process_recording.spawn(parsed.model_dump())
    return {"ok": True, "call_id": call.object_id}


@app.local_entrypoint()
def main(
    kind: Literal["yolo", "sam", "hands", "temporal"],
    media_path: str,
    task: Literal["detect", "instance"] = "detect",
    prompts: str = "",
    output_json: str = "",
    max_frames: int = 240,
    target_fps: float = 10.0,
    conf: float = 0.25,
    imgsz: int = 640,
    vid_stride: int = 1,
) -> None:
    path = Path(media_path).expanduser().resolve()
    media = path.read_bytes()
    suffix = path.suffix or ".mp4"

    if kind == "yolo":
        result = Yolo26().predict.remote(
            media,
            suffix=suffix,
            task=task,
            conf=conf,
            imgsz=imgsz,
            vid_stride=vid_stride,
            max_frames=max_frames,
        )
    elif kind == "sam":
        text_prompts = [prompt.strip() for prompt in prompts.split(",") if prompt.strip()]
        result = SAM31Segmenter().segment.remote(
            media,
            suffix=suffix,
            text_prompts=text_prompts,
            conf=conf,
            imgsz=imgsz,
            max_frames=max_frames,
        )
    elif kind == "hands":
        result = MediaPipeHands().landmarks.remote(
            media,
            suffix=suffix,
            target_fps=target_fps,
            max_frames=max_frames,
        )
    elif kind == "temporal":
        result = TemporalActionSegmenter().segment.remote(
            media,
            suffix=suffix,
            target_fps=target_fps,
            max_segments=max_frames,
        )
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    _upload_output(result, output_json)


@app.function(
    image=orchestrator_image,
    secrets=[backend_secret],
    timeout=120,
)
def list_splat_artifacts(recording_id: str) -> list[dict[str, Any]]:
    """List the files Supabase Storage has under
    ``{recording_id}/analysis/gaussian_splat/``."""
    from backend.artifacts import gaussian_splat_dir as _dir

    api = SupabaseApi(SupabaseConfig.from_service_role_env())
    try:
        prefix = _dir(recording_id)
        res = api.client.post(
            f"{api.config.url}/storage/v1/object/list/recordings",
            headers=api.rest_headers(),
            content=json.dumps({"prefix": prefix, "limit": 100}),
        )
        return api._json(res) or []
    finally:
        api.close()


@app.local_entrypoint()
def list_splat(recording_id: str) -> None:
    """List Supabase Storage objects under the recording's gaussian_splat dir."""
    rows = list_splat_artifacts.remote(recording_id)
    print(json.dumps(rows, indent=2, default=str))


@app.function(
    image=orchestrator_image,
    secrets=[backend_secret],
    timeout=120,
)
def mark_splat_succeeded_remote(recording_id: str) -> dict[str, Any]:
    """Read the already-uploaded manifest and upsert a succeeded job row.

    Useful after a successful train when the DB row write was skipped (e.g.
    the kind-constraint migration was applied later)."""
    from backend.splat.manifest import MANIFEST_VERSION  # noqa: F401

    api = SupabaseApi(SupabaseConfig.from_service_role_env())
    try:
        manifest_path = analysis_artifact_paths(recording_id)["gaussian_splat"]
        manifest_bytes = api.download_bytes("recordings", manifest_path)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        summary = {
            "num_gaussians": manifest["splat"]["num_gaussians"],
            "frame_count": manifest["camera_path"]["frame_count"],
            "fps": manifest["camera_path"]["fps"],
            "train_duration_seconds": manifest["train"]["duration_seconds"],
            "iterations": manifest["train"]["iterations"],
            "gpu": manifest["train"]["gpu"],
        }
        api.upsert_rows(
            "recording_analysis_jobs",
            [
                {
                    "recording_id": recording_id,
                    "kind": "gaussian_splat",
                    "status": "succeeded",
                    "artifact_path": manifest_path,
                    "summary": summary,
                    "error": None,
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                }
            ],
            on_conflict="recording_id,kind",
        )
        return {"manifest_path": manifest_path, "summary": summary}
    finally:
        api.close()


@app.local_entrypoint()
def mark_splat(recording_id: str) -> None:
    """Write a succeeded recording_analysis_jobs row for an already-uploaded splat."""
    result = mark_splat_succeeded_remote.remote(recording_id)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def run_splat(
    recording_id: str,
    storage_path: str = "",
    max_num_iterations: int = 7000,
    output_json: str = "",
    update_db: bool = True,
) -> None:
    """Run gaussian splat training for an existing Supabase recording.

    The recording row, video.mp4, depth.bin, poses.jsonl, and intrinsics.json
    must already exist in the ``recordings`` bucket. Artifacts are uploaded to
    ``{recording_id}/analysis/gaussian_splat/``.

    DB updates to ``recording_analysis_jobs`` happen inside the trainer
    container (which has Supabase creds via the Modal secret). If the
    kind-constraint migration hasn't been applied yet, the DB writes will be
    logged-and-skipped on the container side; storage uploads still complete.
    Pass ``--no-update-db`` to skip DB writes entirely.
    """
    storage = (storage_path or f"{recording_id}/").strip().strip("/") + "/"
    result = SplatfactoTrainer().train.remote(
        recording_id,
        storage,
        max_num_iterations=max_num_iterations,
        update_db=update_db,
    )
    _upload_output(result, output_json)


def _error_message(error: BaseException) -> str:
    message = str(error).strip()
    return message if message else error.__class__.__name__
