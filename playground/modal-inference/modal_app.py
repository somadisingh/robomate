from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import modal

from modal_inference.hand_landmarks import ensure_hand_model, infer_hands
from modal_inference.media import (
    bounded_max_frames,
    is_video_suffix,
    output_json as write_output_json,
    write_media_bytes,
)
from modal_inference.ultralytics_results import result_to_record


APP_NAME = "copilot-hackathon-modal-inference"
MODEL_VOLUME_NAME = "copilot-hackathon-modal-inference-models"
MODEL_ROOT = "/models"

THIS_DIR = Path(__file__).resolve().parent
TAS_SOURCE_DIR = THIS_DIR.parent / "temporal_action_segmentation" / "temporal_action_segmentation"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgles2", "libglib2.0-0", "libgl1")
    .pip_install("numpy>=2.2.0", "opencv-python-headless>=4.13.0.92")
)

yolo_image = base_image.pip_install("ultralytics>=8.4.53").add_local_dir(
    THIS_DIR / "modal_inference",
    remote_path="/root/modal_inference",
)

sam_image = base_image.pip_install(
    "ultralytics>=8.4.53",
    "git+https://github.com/ultralytics/CLIP.git",
).add_local_dir(
    THIS_DIR / "modal_inference",
    remote_path="/root/modal_inference",
)

mediapipe_runtime_image = base_image.pip_install("mediapipe>=0.10.35")

mediapipe_image = mediapipe_runtime_image.add_local_dir(
    THIS_DIR / "modal_inference",
    remote_path="/root/modal_inference",
)

tas_image = mediapipe_runtime_image.add_local_dir(
    THIS_DIR / "modal_inference",
    remote_path="/root/modal_inference",
).add_local_dir(
    TAS_SOURCE_DIR,
    remote_path="/root/temporal_action_segmentation",
    ignore=["**/__pycache__/**", "**/.pytest_cache/**"],
)


def _cuda_device() -> int | str:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _upload_output(payload: object, output_path: str) -> None:
    if output_path:
        write_output_json(Path(output_path).expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


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
    gpu=["L4", "A10", "L40S"],
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
        imgsz: int = 640,
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
            predictor = self._predictor(video=video, checkpoint=checkpoint, conf=conf, imgsz=imgsz)
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

        return {
            "engine": "sam-3.1-ultralytics",
            "task": "instance",
            "model": str(checkpoint),
            "text_prompts": prompts,
            "frame_count": len(frames),
            "frames": frames,
            "settings": {"conf": conf, "imgsz": imgsz},
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

        from temporal_action_segmentation.pipeline import PipelineConfig, process_videos

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
                render_contact_sheets=False,
                write_review=False,
                labeler="none",
                max_segments=max_segments,
            )
            records = process_videos([input_path], config)

        return {
            "engine": "copilot-hackathon-temporal-action-segmentation",
            "model": str(self.model_path),
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
