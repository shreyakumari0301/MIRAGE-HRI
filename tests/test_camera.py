"""Camera-open tests using fake captures. No webcam is required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.camera import open_camera  # noqa: E402
from perception.errors import CameraUnavailableError  # noqa: E402


class _CV2:
    CAP_DSHOW = 700
    CAP_ANY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4


class FakeCapture:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released = True
        self.opened = False

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True


class OpenCameraTests(unittest.TestCase):
    def test_closed_camera_raises_and_releases(self) -> None:
        created: list[FakeCapture] = []

        def factory(index: int, backend: int) -> FakeCapture:
            self.assertEqual(index, 3)
            self.assertEqual(backend, _CV2.CAP_DSHOW)
            capture = FakeCapture(opened=False)
            created.append(capture)
            return capture

        with self.assertRaises(CameraUnavailableError) as caught:
            open_camera(3, 640, 480, "dshow", cv2_module=_CV2(), factory=factory)
        self.assertIn("camera index 3", str(caught.exception))
        self.assertTrue(created[0].released)

    def test_open_camera_sets_frame_size(self) -> None:
        def factory(index: int, backend: int) -> FakeCapture:
            return FakeCapture(opened=True)

        capture = open_camera(0, 640, 480, "any", cv2_module=_CV2(), factory=factory)
        assert isinstance(capture, FakeCapture)
        self.assertEqual(capture.props[_CV2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertEqual(capture.props[_CV2.CAP_PROP_FRAME_HEIGHT], 480)
        self.assertFalse(capture.released)

    def test_unknown_backend_names_the_config_error(self) -> None:
        with self.assertRaises(CameraUnavailableError):
            open_camera(0, 640, 480, "msmf", cv2_module=_CV2(), factory=lambda *_: FakeCapture(True))


if __name__ == "__main__":
    unittest.main()
