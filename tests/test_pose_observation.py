"""Pose status tests that do not need a camera or a model file."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pose_detector import (  # noqa: E402
    PoseDetector,
    observation_from_landmarks,
    observation_from_result,
)
from perception.status import FrameStatus  # noqa: E402


def _point(x: float, y: float, visibility: float | None, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _pose(visible: int, total: int = 12) -> list[SimpleNamespace]:
    points = [_point(0.5, 0.5, 0.9) for _ in range(visible)]
    points.extend(_point(0.5, 0.5, 0.1) for _ in range(total - visible))
    return points


class ObservationTests(unittest.TestCase):
    def test_no_landmarks_is_unknown(self) -> None:
        observation = observation_from_landmarks(
            None, 10, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.UNKNOWN)
        self.assertIsNone(observation.landmarks)
        self.assertEqual(observation.visible_count, 0)

    def test_empty_pose_list_is_unknown(self) -> None:
        result = SimpleNamespace(pose_landmarks=[])
        observation = observation_from_result(
            result, 4, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.UNKNOWN)
        self.assertIsNone(observation.landmarks)

    def test_too_few_visible_landmarks_is_unknown_and_keeps_points(self) -> None:
        observation = observation_from_landmarks(
            _pose(visible=7), 3, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.UNKNOWN)
        self.assertEqual(observation.visible_count, 7)
        self.assertIsNotNone(observation.landmarks)
        assert observation.landmarks is not None
        self.assertEqual(len(observation.landmarks), 12)

    def test_enough_visible_landmarks_is_detected(self) -> None:
        observation = observation_from_landmarks(
            _pose(visible=8), 3, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.DETECTED)
        self.assertEqual(observation.visible_count, 8)

    def test_missing_visibility_is_not_counted(self) -> None:
        points = [_point(0.2, 0.2, None) for _ in range(10)]
        observation = observation_from_landmarks(
            points, 1, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.UNKNOWN)
        self.assertEqual(observation.visible_count, 0)

    def test_non_finite_coordinates_keep_their_index_empty(self) -> None:
        points = _pose(visible=8)
        points.append(_point(float("nan"), 0.2, 0.99))
        observation = observation_from_landmarks(
            points, 1, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.DETECTED)
        assert observation.landmarks is not None
        self.assertEqual(len(observation.landmarks), 13)
        self.assertIsNone(observation.landmarks[-1])

    def test_malformed_result_is_unavailable(self) -> None:
        observation = observation_from_result(
            SimpleNamespace(), 1, min_visibility=0.5, min_visible_landmarks=8
        )
        self.assertIs(observation.status, FrameStatus.UNAVAILABLE)
        self.assertIsNone(observation.landmarks)


class DetectorFailureTests(unittest.TestCase):
    def test_inference_error_is_unavailable_without_landmarks(self) -> None:
        class Boom:
            def detect_for_video(self, image: object, timestamp_ms: int) -> object:
                raise RuntimeError("model blew up")

            def close(self) -> None:
                return None

        detector = PoseDetector(Path("missing.task"), landmarker=Boom(), image_factory=lambda frame: frame)
        frame = _array()
        observation = detector.detect(frame)
        self.assertIs(observation.status, FrameStatus.UNAVAILABLE)
        self.assertIsNone(observation.landmarks)
        self.assertIn("model blew up", observation.message)
        detector.close()

    def test_missing_frame_is_unavailable(self) -> None:
        detector = PoseDetector(Path("missing.task"), landmarker=BoomLandmarker(), image_factory=lambda frame: frame)
        observation = detector.detect(None)
        self.assertIs(observation.status, FrameStatus.UNAVAILABLE)
        self.assertIsNone(observation.landmarks)

    def test_timestamps_increase_when_the_clock_does_not(self) -> None:
        detector = PoseDetector(Path("missing.task"), landmarker=BoomLandmarker(), image_factory=lambda frame: frame)
        detector._clock = lambda: 1.0
        first = detector.next_timestamp_ms()
        second = detector.next_timestamp_ms()
        self.assertGreater(second, first)
        detector.close()


class BoomLandmarker:
    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        raise AssertionError("inference should not run")

    def close(self) -> None:
        return None


def _array() -> object:
    import numpy as np

    return np.zeros((8, 8, 3), dtype="uint8")


if __name__ == "__main__":
    unittest.main()
