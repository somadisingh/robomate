from backend.contracts import RecordingRecord
from backend.orchestrator import MIN_DEPTH_FRAMES_FOR_SPLAT, gaussian_splat_preflight


def _make_recording(**overrides) -> RecordingRecord:
    base = {
        "id": "rec-1",
        "storage_path": "rec-1/",
        "streams": ["video.mp4", "depth.bin"],
    }
    base.update(overrides)
    return RecordingRecord.model_validate(base)


def test_preflight_returns_false_when_depth_metadata_missing():
    rec = _make_recording()
    assert gaussian_splat_preflight(rec) is False


def test_preflight_returns_false_when_depth_partially_set():
    rec = _make_recording(depth_width=320, depth_height=240, depth_frame_count=None)
    assert gaussian_splat_preflight(rec) is False


def test_preflight_returns_false_when_frames_below_threshold():
    rec = _make_recording(
        depth_width=320,
        depth_height=240,
        depth_frame_count=MIN_DEPTH_FRAMES_FOR_SPLAT - 1,
    )
    assert gaussian_splat_preflight(rec) is False


def test_preflight_returns_true_when_sufficient_depth_frames():
    rec = _make_recording(
        depth_width=320,
        depth_height=240,
        depth_frame_count=MIN_DEPTH_FRAMES_FOR_SPLAT,
    )
    assert gaussian_splat_preflight(rec) is True


def test_preflight_returns_true_for_well_populated_recording():
    rec = _make_recording(
        depth_width=320,
        depth_height=240,
        depth_frame_count=600,
    )
    assert gaussian_splat_preflight(rec) is True
