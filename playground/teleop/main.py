from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from teleop.pose_mapper import MappingConfig, TeleopMapper
from teleop.robot_backend import DryRunBackend, SO101Backend, SO101BackendConfig
from teleop.safety import SafetyConfig, TargetFilter
from teleop.tracker import (
    HAND_MODEL_URL,
    LatestHandResult,
    LatestPoseResult,
    POSE_LEFT_ELBOW,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_MODEL_URL,
    POSE_RIGHT_ELBOW,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
    create_hand_landmarker,
    create_pose_landmarker,
    default_hand_model_path,
    default_pose_model_path,
    draw_overlay,
    ensure_model,
    frame_to_mp_image,
    fuse_samples,
    open_camera,
)
from teleop.types import (
    CONTROLLED_KEYS,
    FilterResult,
    FreezeReason,
    RobotTargets,
    TeleopSample,
)


WINDOW_NAME = "Egocentric SO101 Teleop"

CAMERA_DEFAULTS_BY_INDEX = {
    0: (640, 480, 30),
}
FALLBACK_CAMERA_DEFAULTS = (1280, 720, 30)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Egocentric SO101 6-DOF teleoperation from MediaPipe Pose + Hand."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--check", action="store_true")

    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--mirror-hand", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--presence-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-confidence", type=float, default=0.45)
    parser.add_argument("--min-pose-visibility", type=float, default=0.6)

    parser.add_argument("--pose-model-path", type=Path, default=default_pose_model_path())
    parser.add_argument("--hand-model-path", type=Path, default=default_hand_model_path())

    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--robot-port", type=str)
    parser.add_argument("--robot-id", type=str)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("../so101/calibration/robots/so_follower"),
    )
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--deadman-key", type=str, default="")
    parser.add_argument("--deadman-grace-ms", type=int, default=175)

    parser.add_argument("--shoulder-pan-gain", type=float, default=20.0)
    parser.add_argument("--shoulder-lift-gain", type=float, default=20.0)
    parser.add_argument("--elbow-flex-gain", type=float, default=20.0)
    parser.add_argument("--wrist-flex-gain", type=float, default=30.0)
    parser.add_argument("--wrist-roll-gain", type=float, default=60.0)
    parser.add_argument("--gripper-open", type=float, default=80.0)
    parser.add_argument("--gripper-closed", type=float, default=20.0)
    parser.add_argument("--pinch-closed-ratio", type=float, default=0.35)
    parser.add_argument("--pinch-open-ratio", type=float, default=1.40)

    parser.add_argument("--shoulder-pan-limit", type=float, default=20.0)
    parser.add_argument("--shoulder-lift-limit", type=float, default=20.0)
    parser.add_argument("--elbow-flex-limit", type=float, default=25.0)
    parser.add_argument("--wrist-flex-limit", type=float, default=15.0)
    parser.add_argument("--wrist-roll-limit", type=float, default=25.0)
    parser.add_argument("--gripper-min", type=float, default=15.0)
    parser.add_argument("--gripper-max", type=float, default=85.0)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--stale-timeout-ms", type=int, default=200)

    return parser.parse_args(argv)


def apply_camera_defaults(args: argparse.Namespace) -> None:
    width, height, fps = CAMERA_DEFAULTS_BY_INDEX.get(args.camera_index, FALLBACK_CAMERA_DEFAULTS)
    if args.width is None:
        args.width = width
    if args.height is None:
        args.height = height
    if args.fps is None:
        args.fps = fps


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
        "min_pose_visibility",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.gripper_min >= args.gripper_max:
        raise SystemExit("--gripper-min must be less than --gripper-max")

    try:
        MappingConfig(
            shoulder_pan_gain=args.shoulder_pan_gain,
            shoulder_lift_gain=args.shoulder_lift_gain,
            elbow_flex_gain=args.elbow_flex_gain,
            wrist_flex_gain=args.wrist_flex_gain,
            wrist_roll_gain=args.wrist_roll_gain,
            gripper_open=args.gripper_open,
            gripper_closed=args.gripper_closed,
            pinch_closed_ratio=args.pinch_closed_ratio,
            pinch_open_ratio=args.pinch_open_ratio,
            mirror_hand=_resolve_mirror_hand(args),
        )
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


def _resolve_mirror_hand(args: argparse.Namespace) -> bool:
    if args.mirror_hand == "on":
        return True
    if args.mirror_hand == "off":
        return False
    return args.arm == "left"


def validate_robot_port_matches_id(robot_port: str, robot_id: str) -> None:
    port_serial = robot_port_serial_hint(robot_port)
    robot_serial = robot_id_serial_hint(robot_id)
    if port_serial is None or robot_serial is None or port_serial == robot_serial:
        return
    raise SystemExit(
        f"--robot-port appears to be for serial {port_serial!r}, but --robot-id is {robot_id!r}."
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


@dataclass
class LoopState:
    sync_enabled: bool = False
    notice: str | None = None
    send_failed: bool = False


def main() -> None:
    args = parse_args()
    apply_camera_defaults(args)
    pose_model = ensure_model(args.pose_model_path, url=POSE_MODEL_URL)
    hand_model = ensure_model(args.hand_model_path, url=HAND_MODEL_URL)
    validate_args(args)
    if args.check:
        print(f"Pose model ready: {pose_model}")
        print(f"Hand model ready: {hand_model}")
        if args.enable_robot:
            check_robot_imports(args)
            print("Robot config validated.")
        return

    backend = make_backend(args)
    capture = None
    try:
        capture = open_camera(args.camera_index, args.width, args.height)
        backend.connect()
        mapper = TeleopMapper(make_mapping_config(args))
        target_filter = TargetFilter(
            make_safety_config(args, backend.baseline_targets), backend.baseline_targets
        )

        latest_pose = LatestPoseResult()
        latest_hand = LatestHandResult()
        with create_pose_landmarker(
            model_path=pose_model,
            detection_confidence=args.detection_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            result_callback=latest_pose.update,
        ) as pose_landmarker, create_hand_landmarker(
            model_path=hand_model,
            max_hands=args.max_hands,
            detection_confidence=args.detection_confidence,
            presence_confidence=args.presence_confidence,
            tracking_confidence=args.tracking_confidence,
            result_callback=latest_hand.update,
        ) as hand_landmarker:
            run_loop(
                args=args,
                capture=capture,
                pose_landmarker=pose_landmarker,
                hand_landmarker=hand_landmarker,
                latest_pose=latest_pose,
                latest_hand=latest_hand,
                mapper=mapper,
                target_filter=target_filter,
                backend=backend,
            )
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        backend.disconnect()


def check_robot_imports(args) -> None:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir.expanduser().resolve(),
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    _ = SO101Follower


def make_backend(args):
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


def make_mapping_config(args) -> MappingConfig:
    return MappingConfig(
        shoulder_pan_gain=args.shoulder_pan_gain,
        shoulder_lift_gain=args.shoulder_lift_gain,
        elbow_flex_gain=args.elbow_flex_gain,
        wrist_flex_gain=args.wrist_flex_gain,
        wrist_roll_gain=args.wrist_roll_gain,
        gripper_open=args.gripper_open,
        gripper_closed=args.gripper_closed,
        pinch_closed_ratio=args.pinch_closed_ratio,
        pinch_open_ratio=args.pinch_open_ratio,
        mirror_hand=_resolve_mirror_hand(args),
    )


def make_safety_config(args, baseline: RobotTargets) -> SafetyConfig:
    if args.gripper_min >= args.gripper_max:
        raise ValueError("--gripper-min must be less than --gripper-max")
    return SafetyConfig(
        limits={
            "shoulder_pan.pos": (
                baseline.shoulder_pan - args.shoulder_pan_limit,
                baseline.shoulder_pan + args.shoulder_pan_limit,
            ),
            "shoulder_lift.pos": (
                baseline.shoulder_lift - args.shoulder_lift_limit,
                baseline.shoulder_lift + args.shoulder_lift_limit,
            ),
            "elbow_flex.pos": (
                baseline.elbow_flex - args.elbow_flex_limit,
                baseline.elbow_flex + args.elbow_flex_limit,
            ),
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
        max_delta={key: args.max_delta for key in CONTROLLED_KEYS},
        smoothing=args.smoothing,
        stale_timeout_ms=args.stale_timeout_ms,
        min_pose_visibility=args.min_pose_visibility,
        min_hand_confidence=args.min_hand_confidence,
    )


def run_loop(
    *,
    args,
    capture,
    pose_landmarker,
    hand_landmarker,
    latest_pose: LatestPoseResult,
    latest_hand: LatestHandResult,
    mapper: TeleopMapper,
    target_filter: TargetFilter,
    backend,
) -> None:
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
    consecutive_empty_frames = 0
    max_consecutive_empty_frames = 30

    while True:
        loop_start = time.monotonic()
        ok, frame = capture.read()
        if not ok:
            consecutive_empty_frames += 1
            if consecutive_empty_frames > max_consecutive_empty_frames:
                raise RuntimeError(
                    f"Camera returned {consecutive_empty_frames} empty frames in a row."
                )
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            time.sleep(1.0 / max(args.fps, 1))
            continue
        consecutive_empty_frames = 0
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        mp_image = frame_to_mp_image(frame)
        pose_landmarker.detect_async(mp_image, timestamp_ms)
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        arm_sample = latest_pose.best_arm_sample(arm=args.arm)
        hand_samples = latest_hand.all_hand_samples()
        teleop_sample = fuse_samples(
            arm=arm_sample, hands=hand_samples, timestamp_ms=timestamp_ms
        )

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            handle_sync_toggle(state)
        if key == ord("n"):
            target_filter = handle_neutral_capture(
                sample=teleop_sample,
                now_ms=timestamp_ms,
                mapper=mapper,
                target_filter=target_filter,
                baseline_targets=backend.baseline_targets,
                min_pose_visibility=args.min_pose_visibility,
                min_hand_confidence=args.min_hand_confidence,
                stale_timeout_ms=args.stale_timeout_ms,
                state=state,
            )
        if args.deadman_key and key == ord(args.deadman_key):
            last_deadman_ms = timestamp_ms

        deadman_active = not args.deadman_key or timestamp_ms - last_deadman_ms <= args.deadman_grace_ms
        tracking_ok = sample_is_usable(
            teleop_sample,
            now_ms=timestamp_ms,
            min_pose_visibility=args.min_pose_visibility,
            min_hand_confidence=args.min_hand_confidence,
            stale_timeout_ms=args.stale_timeout_ms,
        )
        desired = target_filter.last_targets
        if tracking_ok and mapper.neutral_ready:
            try:
                desired = mapper.map(teleop_sample)
            except ValueError:
                tracking_ok = False

        sample_ts = teleop_sample.timestamp_ms if teleop_sample.arm is not None else None
        try:
            last_result = target_filter.update(
                desired,
                now_ms=timestamp_ms,
                sample_timestamp_ms=sample_ts,
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
        fps_display = (
            instantaneous_fps if fps_display == 0.0 else (0.9 * fps_display) + (0.1 * instantaneous_fps)
        )
        previous_frame_time = now

        status_lines = build_status_lines(
            args=args,
            fps_display=fps_display,
            sync_enabled=state.sync_enabled,
            neutral_ready=mapper.neutral_ready,
            sample=teleop_sample,
            result=last_result,
            notice=state.notice,
        )
        arm_image_landmarks = _arm_image_landmarks(args.arm, teleop_sample)
        draw_overlay(
            frame,
            arm=teleop_sample.arm,
            hand=teleop_sample.hand,
            status_lines=status_lines,
            image_size=(args.width, args.height),
            arm_image_landmarks=arm_image_landmarks,
        )
        cv2.imshow(WINDOW_NAME, frame)

        elapsed = time.monotonic() - loop_start
        sleep_s = max((1.0 / args.fps) - elapsed, 0.0)
        if sleep_s:
            time.sleep(sleep_s)


def _arm_image_landmarks(
    arm: str, sample: TeleopSample
) -> dict[str, tuple[float, float]] | None:
    if sample.arm is None:
        return None
    return {
        "shoulder": sample.arm.shoulder_image_xy,
        "elbow": sample.arm.elbow_image_xy,
        "wrist": sample.arm.wrist_image_xy,
    }


def sample_is_usable(
    sample: TeleopSample,
    *,
    now_ms: int,
    min_pose_visibility: float,
    min_hand_confidence: float,
    stale_timeout_ms: int,
) -> bool:
    if sample.arm is None or sample.hand is None:
        return False
    if sample.arm.shoulder.visibility < min_pose_visibility:
        return False
    if sample.arm.elbow.visibility < min_pose_visibility:
        return False
    if sample.arm.wrist.visibility < min_pose_visibility:
        return False
    if sample.hand.confidence < min_hand_confidence:
        return False
    if now_ms - sample.timestamp_ms > stale_timeout_ms:
        return False
    return True


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
    sample: TeleopSample,
    now_ms: int,
    mapper: TeleopMapper,
    target_filter: TargetFilter,
    baseline_targets: RobotTargets,
    min_pose_visibility: float,
    min_hand_confidence: float,
    stale_timeout_ms: int,
    state: LoopState,
) -> TargetFilter:
    if not sample_is_usable(
        sample,
        now_ms=now_ms,
        min_pose_visibility=min_pose_visibility,
        min_hand_confidence=min_hand_confidence,
        stale_timeout_ms=stale_timeout_ms,
    ):
        state.notice = "neutral rejected: tracking degraded"
        return target_filter

    try:
        mapper.capture_neutral(sample, baseline_targets)
        state.notice = "neutral captured"
        return TargetFilter(target_filter.config, baseline_targets)
    except ValueError as exc:
        state.notice = f"neutral rejected: {exc}"
        return target_filter


def handle_backend_send(
    backend, result: FilterResult, target_filter: TargetFilter, state: LoopState
) -> FilterResult:
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


def build_status_lines(
    *,
    args,
    fps_display: float,
    sync_enabled: bool,
    neutral_ready: bool,
    sample: TeleopSample,
    result: FilterResult,
    notice: str | None,
) -> list[str]:
    robot_state = "ROBOT" if args.enable_robot else "DRY"
    arm_state = "none"
    if sample.arm is not None:
        arm_state = (
            f"{args.arm} s={sample.arm.shoulder.visibility:.2f} "
            f"e={sample.arm.elbow.visibility:.2f} w={sample.arm.wrist.visibility:.2f}"
        )
    hand_state = "none" if sample.hand is None else f"{sample.hand.handedness} {sample.hand.confidence:.2f}"
    clamp_text = ",".join(result.clamped_keys) if result.clamped_keys else "none"
    targets = result.targets
    line1 = (
        f"{robot_state} | {fps_display:4.1f} FPS | sync={'on' if sync_enabled else 'off'} | "
        f"neutral={'yes' if neutral_ready else 'no'} | reason={result.reason.value} | clamp={clamp_text}"
    )
    line2 = f"arm={arm_state} | hand={hand_state}"
    line3 = (
        f"pan={targets.shoulder_pan:.1f} lift={targets.shoulder_lift:.1f} "
        f"elb={targets.elbow_flex:.1f} wf={targets.wrist_flex:.1f} "
        f"wr={targets.wrist_roll:.1f} grip={targets.gripper:.1f}"
    )
    lines = [line1, line2, line3]
    if notice:
        lines.append(f"notice: {notice}")
    return lines


if __name__ == "__main__":
    main()
