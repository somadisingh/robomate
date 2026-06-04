from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
import time
from pathlib import Path

import cv2

from mediapipe_so101.hand_tracker import (
    LatestHandResult,
    create_landmarker,
    default_model_path,
    draw_sample,
    ensure_model,
    frame_to_mp_image,
    open_camera,
)
from mediapipe_so101.pose_mapper import MappingConfig, PoseMapper
from mediapipe_so101.robot_backend import DryRunBackend, SO101Backend, SO101BackendConfig
from mediapipe_so101.safety import SafetyConfig, TargetFilter
from mediapipe_so101.types import FilterResult, FreezeReason, RobotTargets


WINDOW_NAME = "MediaPipe SO101 Wrist Teleop"


@dataclass
class LoopState:
    sync_enabled: bool = False
    notice: str | None = None
    send_failed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate SO101 wrist and gripper from MediaPipe hand pose.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--presence-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-confidence", type=float, default=0.45)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--robot-port", type=str)
    parser.add_argument("--robot-id", type=str)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("../so101/calibration/robots/so_follower"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--deadman-key", type=str, default="")
    parser.add_argument("--deadman-grace-ms", type=int, default=175)
    parser.add_argument("--wrist-flex-gain", type=float, default=30.0)
    parser.add_argument("--wrist-roll-gain", type=float, default=60.0)
    parser.add_argument("--gripper-open", type=float, default=80.0)
    parser.add_argument("--gripper-closed", type=float, default=20.0)
    parser.add_argument("--pinch-closed-ratio", type=float, default=0.35)
    parser.add_argument("--pinch-open-ratio", type=float, default=1.40)
    parser.add_argument("--wrist-flex-limit", type=float, default=25.0)
    parser.add_argument("--wrist-roll-limit", type=float, default=45.0)
    parser.add_argument("--gripper-min", type=float, default=15.0)
    parser.add_argument("--gripper-max", type=float, default=85.0)
    parser.add_argument("--max-delta", type=float, default=4.0)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--stale-timeout-ms", type=int, default=150)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_model(args.model_path)
    validate_args(args)
    if args.check:
        print(f"MediaPipe model ready: {model_path}")
        if args.enable_robot:
            check_robot_imports(args)
            print("Robot config validated.")
        return

    backend = make_backend(args)
    capture = None
    try:
        capture = open_camera(args.camera_index, args.width, args.height)
        backend.connect()
        mapper = PoseMapper(
            MappingConfig(
                wrist_flex_gain=args.wrist_flex_gain,
                wrist_roll_gain=args.wrist_roll_gain,
                gripper_open=args.gripper_open,
                gripper_closed=args.gripper_closed,
                pinch_closed_ratio=args.pinch_closed_ratio,
                pinch_open_ratio=args.pinch_open_ratio,
            )
        )
        target_filter = TargetFilter(make_safety_config(args, backend.baseline_targets), backend.baseline_targets)
        latest = LatestHandResult()
        with create_landmarker(
            model_path=model_path,
            max_hands=args.max_hands,
            detection_confidence=args.detection_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            result_callback=latest.update,
        ) as landmarker:
            run_loop(args, capture, landmarker, latest, mapper, target_filter, backend)
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        backend.disconnect()


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.max_hands <= 0:
        raise SystemExit("--max-hands must be positive")
    if args.deadman_key and len(args.deadman_key) != 1:
        raise SystemExit("--deadman-key must be a single character")
    if args.deadman_grace_ms < 0:
        raise SystemExit("--deadman-grace-ms must be non-negative")

    for name in (
        "detection_confidence",
        "presence_confidence",
        "tracking_confidence",
        "min_hand_confidence",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    try:
        MappingConfig(
            wrist_flex_gain=args.wrist_flex_gain,
            wrist_roll_gain=args.wrist_roll_gain,
            gripper_open=args.gripper_open,
            gripper_closed=args.gripper_closed,
            pinch_closed_ratio=args.pinch_closed_ratio,
            pinch_open_ratio=args.pinch_open_ratio,
        )
        make_safety_config(args, RobotTargets(0.0, 0.0, args.gripper_open))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.enable_robot:
        if not args.robot_port:
            raise SystemExit("--robot-port is required with --enable-robot")
        if not args.robot_id:
            raise SystemExit("--robot-id is required with --enable-robot")
        validate_robot_port_matches_id(args.robot_port, args.robot_id)
        calibration_file = args.calibration_dir.expanduser().resolve() / f"{args.robot_id}.json"
        if not calibration_file.exists():
            raise SystemExit(f"Calibration file not found: {calibration_file}")
        if not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0:
            raise SystemExit("--max-relative-target must be finite and positive")


def validate_robot_port_matches_id(robot_port: str, robot_id: str) -> None:
    port_serial = robot_port_serial_hint(robot_port)
    robot_serial = robot_id_serial_hint(robot_id)
    if port_serial is None or robot_serial is None or port_serial == robot_serial:
        return

    raise SystemExit(
        f"--robot-port appears to be for serial {port_serial!r}, but --robot-id is {robot_id!r}. "
        f"Use --robot-id so101_{port_serial} or the matching robot port; refusing before calibration "
        "can be written to the wrong arm."
    )


def robot_port_serial_hint(robot_port: str) -> str | None:
    match = re.search(r"(?:^|[.])(?:usbmodem|usbserial)([-_A-Za-z0-9]+)$", Path(robot_port).name)
    if match is None:
        return None
    return match.group(1).lstrip("-_") or None


def robot_id_serial_hint(robot_id: str) -> str | None:
    prefix = "so101_"
    if not robot_id.startswith(prefix):
        return None
    serial = robot_id.removeprefix(prefix)
    return serial or None


def check_robot_imports(args: argparse.Namespace) -> None:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    _robot_config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir.expanduser().resolve(),
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    _robot_class = SO101Follower


def make_backend(args: argparse.Namespace):
    if not args.enable_robot:
        return DryRunBackend(default_gripper=args.gripper_open)
    return SO101Backend(
        SO101BackendConfig(
            port=args.robot_port,
            robot_id=args.robot_id,
            calibration_dir=args.calibration_dir.expanduser().resolve(),
            max_relative_target=args.max_relative_target,
        )
    )


def make_safety_config(args: argparse.Namespace, baseline: RobotTargets) -> SafetyConfig:
    if args.gripper_min >= args.gripper_max:
        raise ValueError("--gripper-min must be less than --gripper-max")

    return SafetyConfig(
        limits={
            "wrist_flex.pos": (
                baseline.wrist_flex - args.wrist_flex_limit,
                baseline.wrist_flex + args.wrist_flex_limit,
            ),
            "wrist_roll.pos": (
                baseline.wrist_roll - args.wrist_roll_limit,
                baseline.wrist_roll + args.wrist_roll_limit,
            ),
            "gripper.pos": (
                min(args.gripper_min, baseline.gripper),
                max(args.gripper_max, baseline.gripper),
            ),
        },
        max_delta={
            "wrist_flex.pos": args.max_delta,
            "wrist_roll.pos": args.max_delta,
            "gripper.pos": args.max_delta,
        },
        smoothing=args.smoothing,
        stale_timeout_ms=args.stale_timeout_ms,
    )


def run_loop(args, capture, landmarker, latest, mapper, target_filter, backend) -> None:
    start_time = time.monotonic()
    previous_timestamp_ms = -1
    previous_frame_time = start_time
    fps_display = 0.0
    state = LoopState()
    last_deadman_ms = -1
    last_result = FilterResult(
        target_filter.last_targets,
        frozen=True,
        clamped_keys=(),
        reason=FreezeReason.PAUSED,
    )

    while True:
        loop_start = time.monotonic()
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera returned an empty frame.")
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms
        landmarker.detect_async(frame_to_mp_image(frame), timestamp_ms)

        sample = latest.best_sample()
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            handle_sync_toggle(state)
        if key == ord("n"):
            target_filter = handle_neutral_capture(
                sample=sample,
                now_ms=timestamp_ms,
                mapper=mapper,
                target_filter=target_filter,
                baseline_targets=backend.baseline_targets,
                min_hand_confidence=args.min_hand_confidence,
                stale_timeout_ms=target_filter.config.stale_timeout_ms,
                state=state,
            )
        if args.deadman_key and key == ord(args.deadman_key):
            last_deadman_ms = timestamp_ms

        deadman_active = not args.deadman_key or timestamp_ms - last_deadman_ms <= args.deadman_grace_ms
        tracking_ok = sample_is_usable(
            sample,
            now_ms=timestamp_ms,
            min_hand_confidence=args.min_hand_confidence,
            stale_timeout_ms=target_filter.config.stale_timeout_ms,
        )
        desired = target_filter.last_targets
        if sample is not None and mapper.neutral_ready:
            try:
                desired = mapper.map(sample)
            except ValueError:
                tracking_ok = False

        sample_timestamp_ms = sample.timestamp_ms if sample is not None else None
        try:
            last_result = target_filter.update(
                desired,
                now_ms=timestamp_ms,
                sample_timestamp_ms=sample_timestamp_ms,
                sync_enabled=state.sync_enabled,
                neutral_ready=mapper.neutral_ready,
                deadman_active=deadman_active,
                tracking_ok=tracking_ok,
            )
        except ValueError:
            last_result = FilterResult(
                target_filter.last_targets,
                frozen=True,
                clamped_keys=(),
                reason=FreezeReason.TRACKING_LOST,
            )
        last_result = handle_backend_send(backend, last_result, target_filter, state)

        now = time.monotonic()
        instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
        fps_display = instantaneous_fps if fps_display == 0.0 else (0.9 * fps_display) + (0.1 * instantaneous_fps)
        previous_frame_time = now

        draw_sample(frame, sample)
        draw_status(frame, args, fps_display, state.sync_enabled, mapper.neutral_ready, sample, last_result, state.notice)
        cv2.imshow(WINDOW_NAME, frame)

        elapsed = time.monotonic() - loop_start
        sleep_s = max((1.0 / args.fps) - elapsed, 0.0)
        if sleep_s:
            time.sleep(sleep_s)


def sample_is_usable(
    sample,
    *,
    now_ms: int,
    min_hand_confidence: float,
    stale_timeout_ms: int,
) -> bool:
    return neutral_rejection_reason(
        sample,
        now_ms=now_ms,
        min_hand_confidence=min_hand_confidence,
        stale_timeout_ms=stale_timeout_ms,
    ) is None


def neutral_rejection_reason(
    sample,
    *,
    now_ms: int,
    min_hand_confidence: float,
    stale_timeout_ms: int,
) -> FreezeReason | None:
    if sample is None:
        return FreezeReason.TRACKING_LOST
    if sample.confidence < min_hand_confidence:
        return FreezeReason.TRACKING_LOST
    if now_ms - sample.timestamp_ms > stale_timeout_ms:
        return FreezeReason.STALE_RESULT
    return None


def handle_sync_toggle(state: LoopState) -> None:
    if state.send_failed:
        state.sync_enabled = False
        state.notice = "sync locked off: send failed"
        return

    state.sync_enabled = not state.sync_enabled
    if state.sync_enabled:
        state.notice = None


def handle_neutral_capture(
    *,
    sample,
    now_ms: int,
    mapper,
    target_filter: TargetFilter,
    baseline_targets: RobotTargets,
    min_hand_confidence: float,
    stale_timeout_ms: int,
    state: LoopState,
) -> TargetFilter:
    reason = neutral_rejection_reason(
        sample,
        now_ms=now_ms,
        min_hand_confidence=min_hand_confidence,
        stale_timeout_ms=stale_timeout_ms,
    )
    if reason is not None:
        state.notice = f"neutral rejected: {reason.value}"
        return target_filter

    try:
        mapper.capture_neutral(sample, baseline_targets)
        state.notice = "neutral captured"
        return TargetFilter(target_filter.config, baseline_targets)
    except ValueError as exc:
        state.notice = f"neutral rejected: {exc}"
        return target_filter


def handle_backend_send(backend, result: FilterResult, target_filter: TargetFilter, state: LoopState) -> FilterResult:
    if state.send_failed:
        return FilterResult(
            target_filter.last_targets,
            frozen=True,
            clamped_keys=(),
            reason=FreezeReason.PAUSED,
        )
    if result.frozen:
        return result

    try:
        backend.send(result.targets)
        return result
    except Exception as exc:
        state.sync_enabled = False
        state.send_failed = True
        state.notice = f"send failed: {exc}"
        return FilterResult(
            target_filter.last_targets,
            frozen=True,
            clamped_keys=(),
            reason=FreezeReason.PAUSED,
        )


def draw_status(
    frame,
    args,
    fps_display,
    sync_enabled,
    neutral_ready,
    sample,
    result: FilterResult,
    notice: str | None = None,
) -> None:
    robot_state = "ROBOT" if args.enable_robot else "DRY"
    hand_state = "none" if sample is None else f"{sample.handedness} {sample.confidence:.2f}"
    clamp_text = ",".join(result.clamped_keys) if result.clamped_keys else "none"
    status = (
        f"{robot_state} | {fps_display:4.1f} FPS | sync={'on' if sync_enabled else 'off'} | "
        f"neutral={'yes' if neutral_ready else 'no'} | hand={hand_state} | "
        f"reason={result.reason.value} | clamp={clamp_text} | "
        f"flex={result.targets.wrist_flex:.1f} roll={result.targets.wrist_roll:.1f} grip={result.targets.gripper:.1f}"
    )
    if notice:
        status = f"{status} | notice={notice}"
    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
