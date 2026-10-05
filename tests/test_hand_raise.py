"""Synthetic landmark tests for the raw hand-raise rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.draw import render_frame  # noqa: E402
from perception.hand_raise import (  # noqa: E402
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    HandRaiseLabel,
    predict_hand_raise,
    visibility_line,
)
from perception.status import FrameStatus, Landmark, PoseObservation  # noqa: E402

MARGIN = 0.08
VISIBILITY = 0.5


def _pose() -> list[Landmark | None]:
    return [Landmark(0.5, 0.55, 0.0, 0.2) for _ in range(33)]


def _point(y: float, visibility: float | None, x: float = 0.5) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=visibility)


def _predict(landmarks: tuple[Landmark | None, ...] | None) -> object:
    return predict_hand_raise(
        landmarks,
        min_visibility=VISIBILITY,
        raise_margin=MARGIN,
    )


class HandRaiseTests(unittest.TestCase):
    def test_right_wrist_above_shoulder_is_raised(self) -> None:
        points = _pose()
        points[RIGHT_SHOULDER] = _point(0.50, 0.91)
        points[RIGHT_WRIST] = _point(0.20, 0.87)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.RAISED)
        self.assertEqual(prediction.cue, "hand_raised")
        self.assertIn("right", prediction.message)
        self.assertAlmostEqual(prediction.sides[1].clearance, 0.30)
        self.assertNotIn("probability", prediction.message)

    def test_both_wrists_below_shoulders_is_not_raised(self) -> None:
        points = _pose()
        points[LEFT_SHOULDER] = _point(0.40, 0.95)
        points[LEFT_WRIST] = _point(0.70, 0.93)
        points[RIGHT_SHOULDER] = _point(0.42, 0.96)
        points[RIGHT_WRIST] = _point(0.75, 0.90)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.NOT_RAISED)
        self.assertFalse(prediction.sides[0].raised)
        self.assertFalse(prediction.sides[1].raised)

    def test_clearance_below_the_margin_is_not_raised(self) -> None:
        points = _pose()
        points[RIGHT_SHOULDER] = _point(0.50, 0.90)
        points[RIGHT_WRIST] = _point(0.43, 0.90)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.NOT_RAISED)
        self.assertAlmostEqual(prediction.sides[1].clearance, 0.07)

    def test_clearance_equal_to_the_margin_is_raised(self) -> None:
        points = _pose()
        points[LEFT_SHOULDER] = _point(0.50, 0.90)
        points[LEFT_WRIST] = _point(0.42, 0.90)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.RAISED)
        self.assertIn("left", prediction.message)

    def test_missing_pose_is_insufficient(self) -> None:
        prediction = _predict(None)
        self.assertIs(prediction.label, HandRaiseLabel.INSUFFICIENT)
        self.assertIsNone(prediction.sides[0].wrist_visibility)
        self.assertIsNone(prediction.sides[1].shoulder_visibility)

    def test_short_landmark_list_is_insufficient(self) -> None:
        prediction = _predict(tuple(_pose()[:10]))
        self.assertIs(prediction.label, HandRaiseLabel.INSUFFICIENT)

    def test_low_visibility_is_insufficient_and_does_not_count_as_raised(self) -> None:
        points = _pose()
        points[RIGHT_SHOULDER] = _point(0.50, 0.20)
        points[RIGHT_WRIST] = _point(0.10, 0.95)
        points[LEFT_SHOULDER] = _point(0.50, None)
        points[LEFT_WRIST] = _point(0.10, 0.95)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.INSUFFICIENT)
        self.assertFalse(prediction.sides[0].raised)
        self.assertFalse(prediction.sides[1].raised)
        self.assertIsNone(prediction.sides[1].clearance)

    def test_one_usable_lowered_hand_is_not_raised(self) -> None:
        points = _pose()
        points[LEFT_SHOULDER] = _point(0.40, 0.90)
        points[LEFT_WRIST] = _point(0.70, 0.90)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.NOT_RAISED)
        self.assertTrue(prediction.sides[0].usable)
        self.assertFalse(prediction.sides[1].usable)

    def test_point_outside_the_frame_is_not_used(self) -> None:
        points = _pose()
        points[RIGHT_SHOULDER] = _point(0.40, 0.90)
        points[RIGHT_WRIST] = _point(-0.05, 0.90)
        prediction = _predict(tuple(points))
        self.assertIs(prediction.label, HandRaiseLabel.INSUFFICIENT)

    def test_overlay_shows_the_raw_label_and_visibility(self) -> None:
        points = _pose()
        points[RIGHT_SHOULDER] = _point(0.50, 0.91, x=0.30)
        points[RIGHT_WRIST] = _point(0.20, 0.87, x=0.30)
        prediction = _predict(tuple(points))
        text = visibility_line(prediction)
        self.assertIn("hand_raised: raised", text)
        self.assertIn("R w0.87 s0.91", text)
        self.assertIn("L w0.20 s0.20", text)

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        observation = PoseObservation(
            status=FrameStatus.DETECTED,
            landmarks=tuple(points),
            visible_count=4,
            message="Pose landmarks visible (4).",
            timestamp_ms=1,
        )
        image = render_frame(frame, observation, min_visibility=VISIBILITY, hand_raise=prediction)
        self.assertEqual(image.shape, frame.shape)
        wrist_y = int(0.20 * 120)
        wrist_x = int(0.30 * 160)
        ring = image[wrist_y, wrist_x + 8]
        self.assertGreater(int(ring[1]), 150)
        self.assertLess(int(ring[2]), 120)


if __name__ == "__main__":
    unittest.main()
