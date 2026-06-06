from backend.artifacts import (
    ANALYSIS_FILENAMES,
    analysis_artifact_paths,
    detected_object_summary,
    gaussian_splat_dir,
    normalize_sam_prompts,
)


def test_analysis_artifact_paths_are_stable():
    paths = analysis_artifact_paths("abc123")

    assert paths == {
        "gemini_eval": "abc123/analysis/gemini-eval.json",
        "mediapipe_hands": "abc123/analysis/mediapipe-hands.json",
        "yolo_objects": "abc123/analysis/yolo-detections.json",
        "sam_segments": "abc123/analysis/sam-segments.json",
        "temporal_actions": "abc123/analysis/temporal-actions.json",
        "gaussian_splat": "abc123/analysis/gaussian_splat/manifest.json",
        "gemini_temporal_annotations": "abc123/analysis/gemini-temporal-annotations.json",
    }
    assert set(paths) == set(ANALYSIS_FILENAMES)


def test_gaussian_splat_dir_strips_slashes():
    assert gaussian_splat_dir("abc123") == "abc123/analysis/gaussian_splat"
    assert gaussian_splat_dir("/abc123/") == "abc123/analysis/gaussian_splat"


def test_detected_object_summary_compacts_frame_records():
    yolo_payload = {
        "frames": [
            {
                "frame_index": 0,
                "instances": [
                    {"class_name": "cup", "confidence": 0.8},
                    {"class_name": "cup", "confidence": 0.9},
                ],
            },
            {
                "frame_index": 8,
                "instances": [
                    {"class_name": "bottle", "confidence": 0.4},
                ],
            },
        ]
    }

    assert detected_object_summary(yolo_payload) == [
        {
            "class_name": "cup",
            "count": 2,
            "max_confidence": 0.9,
            "representative_frame": 0,
        },
        {
            "class_name": "bottle",
            "count": 1,
            "max_confidence": 0.4,
            "representative_frame": 8,
        },
    ]


def test_normalize_sam_prompts_always_includes_human_hand():
    prompts = normalize_sam_prompts(["Cup", " cup ", "", "can"])

    assert prompts == ["cup", "can", "human hand"]
