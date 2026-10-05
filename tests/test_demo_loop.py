"""Demo loop tests: missing landmarks and a dead camera do not raise."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_loop import run_demo  # noqa: E402
from perception.status import FrameStatus, PoseObservation  # noqa: E402


class FakeCapture:
    def __init__(self, frames: list[object | None]) -> None:
        self.frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> bool:
        return True


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.05
        return self.value


def _observation(status: FrameStatus, message: str) -> PoseObservation:
    return PoseObservation(
        status=status,
        landmarks=None,
        visible_count=0,
        message=message,
        timestamp_ms=0,
    )


class DemoLoopTests(unittest.TestCase):
    def test_unknown_frames_keep_the_loop_running(self) -> None:
        frames = ["a", "b", "c"]
        statuses = [
            _observation(FrameStatus.UNKNOWN, "No pose landmarks in frame."),
            _observation(FrameStatus.DETECTED, "Pose landmarks visible (10)."),
            _observation(FrameStatus.UNKNOWN, "No pose landmarks in frame."),
        ]

        def detect(frame: object) -> PoseObservation:
            return statuses.pop(0)

        rendered: list[object] = []
        summary = run_demo(
            FakeCapture(frames),
            detect,
            lambda frame, observation, fps: rendered.append(frame) or frame,
            max_frames=3,
            max_consecutive_read_failures=30,
            clock=Clock(),
        )
        self.assertEqual(summary.exit_code, 0)
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.unknown, 2)
        self.assertEqual(summary.detected, 1)
        self.assertEqual(summary.unavailable, 0)
        self.assertEqual(rendered, ["a", "b", "c"])

    def test_repeated_read_failures_stop_without_raising(self) -> None:
        summary = run_demo(
            FakeCapture([None, None, None, None]),
            lambda frame: (_ for _ in ()).throw(AssertionError("no frame")),
            lambda frame, observation, fps: frame,
            max_frames=10,
            max_consecutive_read_failures=3,
            clock=Clock(),
        )
        self.assertEqual(summary.exit_code, 1)
        self.assertEqual(summary.frames, 0)
        self.assertGreaterEqual(summary.unavailable, 3)
        self.assertTrue(any("unreadable" in message for message in summary.messages))

    def test_quit_key_ends_the_loop(self) -> None:
        summary = run_demo(
            FakeCapture(["a", "b"]),
            lambda frame: _observation(FrameStatus.UNKNOWN, "No pose landmarks in frame."),
            lambda frame, observation, fps: frame,
            max_frames=None,
            max_consecutive_read_failures=5,
            show=lambda image: None,
            wait_key=lambda: ord("q"),
            clock=Clock(),
        )
        self.assertEqual(summary.exit_code, 0)
        self.assertEqual(summary.frames, 1)
        self.assertIn("Stopped.", summary.messages)

    def test_should_stop_ends_the_loop(self) -> None:
        seen = {"frames": 0}

        def should_stop() -> bool:
            return seen["frames"] >= 1

        def detect(frame: object) -> PoseObservation:
            seen["frames"] += 1
            return _observation(FrameStatus.UNKNOWN, "No pose landmarks in frame.")

        summary = run_demo(
            FakeCapture(["a", "b", "c"]),
            detect,
            lambda frame, observation, fps: frame,
            max_frames=None,
            max_consecutive_read_failures=5,
            clock=Clock(),
            should_stop=should_stop,
        )
        self.assertEqual(summary.frames, 1)
        self.assertIn("Stopped.", summary.messages)


if __name__ == "__main__":
    unittest.main()
