"""Phase 1 webcam demo: show body landmarks and a frame status.

Quit with q or Esc. This demo does not detect hand raises or emit events.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo_loop import ensure_live_camera, run_demo
from perception.camera import open_camera
from perception.draw import render_frame
from perception.errors import MirageError
from perception.pose_detector import PoseDetector
from settings import AppSettings, load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIRAGE-HRI Phase 1 landmark demo")
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
    detector: PoseDetector | None = None
    window = settings.window_name

    def show(image: object) -> None:
        cv2.imshow(window, image)

    def wait_key() -> int:
        return int(cv2.waitKey(1) & 0xFF)

    def render(frame: object, observation: object, fps: float | None) -> object:
        return render_frame(
            frame,
            observation,
            min_visibility=settings.pose.min_landmark_visibility,
            fps=fps,
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
            detector.detect,
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
        if not args.headless:
            cv2.destroyAllWindows()

    for message in summary.messages:
        print(message)
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
