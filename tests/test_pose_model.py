"""One real model inference on a blank image. Skipped if the model file is absent."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pose_detector import PoseDetector  # noqa: E402
from perception.status import FrameStatus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "pose_landmarker_lite.task"


@unittest.skipUnless(MODEL.is_file(), "pose model has not been downloaded")
class BlankFrameModelTests(unittest.TestCase):
    def test_blank_frame_is_unknown_and_has_no_landmarks(self) -> None:
        detector = PoseDetector(MODEL)
        try:
            observation = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        finally:
            detector.close()
        self.assertIs(observation.status, FrameStatus.UNKNOWN)
        self.assertIsNone(observation.landmarks)
        self.assertEqual(observation.visible_count, 0)


if __name__ == "__main__":
    unittest.main()
