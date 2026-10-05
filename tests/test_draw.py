"""Drawing tests. They use an in-memory image, not a window."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.draw import render_frame  # noqa: E402
from perception.status import FrameStatus, Landmark, PoseObservation  # noqa: E402


def _observation(
    status: FrameStatus,
    landmarks: tuple[Landmark, ...] | None,
    message: str,
) -> PoseObservation:
    return PoseObservation(
        status=status,
        landmarks=landmarks,
        visible_count=0 if not landmarks else len(landmarks),
        message=message,
        timestamp_ms=1,
    )


class DrawTests(unittest.TestCase):
    def test_unknown_without_landmarks_draws_a_banner(self) -> None:
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        image = render_frame(
            frame,
            _observation(FrameStatus.UNKNOWN, None, "No pose landmarks in frame."),
            min_visibility=0.5,
        )
        self.assertEqual(image.shape, frame.shape)
        self.assertTrue(np.all(image[70, 2] == np.array([20, 20, 20])))
        self.assertTrue(np.all(frame[40, 60] == 255))

    def test_low_visibility_points_are_not_drawn(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        hidden = Landmark(x=0.5, y=0.25, z=0.0, visibility=0.1)
        image = render_frame(
            frame,
            _observation(FrameStatus.UNKNOWN, (hidden,), "Insufficient landmark visibility (0/8)."),
            min_visibility=0.5,
        )
        self.assertTrue(np.all(image[20, 60] == 0))


if __name__ == "__main__":
    unittest.main()
