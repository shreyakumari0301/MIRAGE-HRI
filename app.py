"""Webcam demo: landmarks, raw hand-raise labels, and filtered robot events.

Quit with q or Esc. The robot hears one event after a raise has held still,
not one event per raised frame.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from datetime import datetime, timezone

from demo_loop import ensure_live_camera, run_demo
from events.consumer import RobotConsumer
from events.log import JsonlLog
from events.schema import raw_frame_record
from events.temporal_filter import TemporalFilter
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


def run(
    settings: AppSettings,
    args: argparse.Namespace,
    *,
    should_stop: Callable[[], bool] | None = None,
    prompt: str | Callable[[], str] | None = None,
) -> int:
    import cv2

    _silence_opencv_logs(cv2)
    camera_index = settings.camera.index if args.camera is None else args.camera
    print("Loading the pose model. This takes a few seconds...", flush=True)
    detector = PoseDetector(
        settings.pose.model_path,
        num_poses=settings.pose.num_poses,
        min_pose_detection_confidence=settings.pose.min_pose_detection_confidence,
        min_pose_presence_confidence=settings.pose.min_pose_presence_confidence,
        min_tracking_confidence=settings.pose.min_tracking_confidence,
        min_landmark_visibility=settings.pose.min_landmark_visibility,
        min_visible_landmarks=settings.pose.min_visible_landmarks,
    )
    print(f"Opening camera {camera_index}...", flush=True)
    capture = None
    try:
        capture = open_camera(
            camera_index,
            settings.camera.width,
            settings.camera.height,
            settings.camera.backend,
            cv2_module=cv2,
        )
        log = JsonlLog(settings.events.log_path)
    except Exception:
        if capture is not None:
            capture.release()
        detector.close()
        raise
    window = settings.window_name
    latest: dict[str, object] = {}
    hand_counts = {label: 0 for label in HandRaiseLabel}
    hand_messages: list[str] = []
    last_hand_label: HandRaiseLabel | None = None
    consumer = RobotConsumer()
    gesture = TemporalFilter(
        settings.filter.activation_ms,
        settings.filter.release_ms,
        settings.filter.cooldown_ms,
    )
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
        update = gesture.update(
            observation,
            prediction,
            source=settings.events.source,
            clock=clock,
        )
        latest["filter_line"] = update.overlay
        filtered = update.to_log()
        if filtered is not None:
            log.write(filtered)
        if update.event is not None:
            consumer.receive(update.event)
        return observation

    def render(frame: object, observation: object, fps: float | None) -> object:
        prediction = latest.get("hand_raise")
        image = render_frame(
            frame,
            observation,
            min_visibility=settings.pose.min_landmark_visibility,
            fps=fps,
            hand_raise=prediction if isinstance(prediction, HandRaisePrediction) else None,
            robot_line=consumer.overlay_line,
            filter_line=latest.get("filter_line") if isinstance(latest.get("filter_line"), str) else None,
        )
        line = prompt() if callable(prompt) else prompt
        if line and getattr(image, "ndim", 0) == 3:
            cv2.rectangle(image, (0, 0), (image.shape[1], 28), (20, 20, 20), -1)
            cv2.putText(
                image,
                line[:90],
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        return image

    try:
        ensure_live_camera(capture, camera_index)
        print(
            "Running. The MIRAGE-HRI window is the live view. Click it and press q to quit.",
            flush=True,
        )
        summary = run_demo(
            capture,
            detect,
            render,
            max_frames=args.frames,
            max_consecutive_read_failures=settings.camera.max_consecutive_read_failures,
            show=None if args.headless else show,
            wait_key=None if args.headless else wait_key,
            should_stop=should_stop,
        )
    finally:
        capture.release()
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
