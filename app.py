"""Webcam demo: landmarks, a raw hand-raise label, and robot events.

Quit with q or Esc. Every raised frame is sent to the simulated robot.
Duplicate triggers are not filtered yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datetime import datetime, timezone

from demo_loop import ensure_live_camera, run_demo
from events.consumer import RobotConsumer
from events.log import JsonlLog
from events.schema import build_interaction_event, raw_frame_record
from perception.camera import open_camera
from perception.draw import render_frame
from perception.errors import MirageError
from perception.hand_raise import HandRaiseLabel, HandRaisePrediction, predict_hand_raise
from perception.pose_detector import PoseDetector
from settings import AppSettings, load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIRAGE-HRI landmark and hand-raise demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--camera", type=int, default=None, help="Override camera index")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Stop after this many successfully read frames",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process frames without opening a window",
    )
    return parser.parse_args(argv)


def _silence_opencv_logs(cv2_module: object) -> None:
    utils = getattr(cv2_module, "utils", None)
    logging = getattr(utils, "logging", None) if utils is not None else None
    if logging is not None and hasattr(logging, "setLogLevel"):
        level = getattr(logging, "LOG_LEVEL_ERROR", 0)
        logging.setLogLevel(level)


def run(settings: AppSettings, args: argparse.Namespace) -> int:
    import cv2

    _silence_opencv_logs(cv2)
    camera_index = settings.camera.index if args.camera is None else args.camera
    capture = open_camera(
        camera_index,
        settings.camera.width,
        settings.camera.height,
        settings.camera.backend,
        cv2_module=cv2,
    )
    try:
        log = JsonlLog(settings.events.log_path)
    except OSError:
        capture.release()
        raise
    detector: PoseDetector | None = None
    window = settings.window_name
    latest: dict[str, object] = {}
    hand_counts = {label: 0 for label in HandRaiseLabel}
    hand_messages: list[str] = []
    last_hand_label: HandRaiseLabel | None = None
    consumer = RobotConsumer()
    clock = lambda: datetime.now(timezone.utc)

    def show(image: object) -> None:
        cv2.imshow(window, image)

    def wait_key() -> int:
        return int(cv2.waitKey(1) & 0xFF)

    def detect(frame: object) -> object:
        nonlocal last_hand_label
        assert detector is not None
        observation = detector.detect(frame)
        prediction = predict_hand_raise(
            observation.landmarks,
            min_visibility=settings.pose.min_landmark_visibility,
            raise_margin=settings.hand_raise.raise_margin,
        )
        latest["hand_raise"] = prediction
        hand_counts[prediction.label] += 1
        if prediction.label is not last_hand_label:
            hand_messages.append(prediction.message)
            last_hand_label = prediction.label
        log.write(raw_frame_record(observation, prediction, clock=clock).to_dict())
        event = build_interaction_event(
            observation,
            prediction,
            source=settings.events.source,
            clock=clock,
        )
        if event is not None:
            log.write(event.to_dict())
            consumer.receive(event)
        return observation

    def render(frame: object, observation: object, fps: float | None) -> object:
        prediction = latest.get("hand_raise")
        return render_frame(
            frame,
            observation,
            min_visibility=settings.pose.min_landmark_visibility,
            fps=fps,
            hand_raise=prediction if isinstance(prediction, HandRaisePrediction) else None,
            robot_line=consumer.overlay_line,
        )

    try:
        detector = PoseDetector(
            settings.pose.model_path,
            num_poses=settings.pose.num_poses,
            min_pose_detection_confidence=settings.pose.min_pose_detection_confidence,
            min_pose_presence_confidence=settings.pose.min_pose_presence_confidence,
            min_tracking_confidence=settings.pose.min_tracking_confidence,
            min_landmark_visibility=settings.pose.min_landmark_visibility,
            min_visible_landmarks=settings.pose.min_visible_landmarks,
        )
        ensure_live_camera(capture, camera_index)
        summary = run_demo(
            capture,
            detect,
            render,
            max_frames=args.frames,
            max_consecutive_read_failures=settings.camera.max_consecutive_read_failures,
            show=None if args.headless else show,
            wait_key=None if args.headless else wait_key,
        )
    finally:
        capture.release()
        if detector is not None:
            detector.close()
        log.close()
        if not args.headless:
            cv2.destroyAllWindows()

    for message in summary.messages:
        print(message)
    for message in hand_messages:
        print(message)
    print(
        "hand_raise "
        f"raised={hand_counts[HandRaiseLabel.RAISED]} "
        f"not_raised={hand_counts[HandRaiseLabel.NOT_RAISED]} "
        f"insufficient={hand_counts[HandRaiseLabel.INSUFFICIENT]}"
    )
    print(f"robot_events={len(consumer.events)} log={log.path}")
    print(
        "summary "
        f"frames={summary.frames} "
        f"detected={summary.detected} "
        f"unknown={summary.unknown} "
        f"unavailable={summary.unavailable} "
        f"elapsed_s={summary.elapsed_s:.2f} "
        f"fps={summary.fps:.1f}"
    )
    return summary.exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings(args.config)
        return run(settings, args)
    except MirageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
