"""Webcam open and read helpers.

OpenCV's default Windows backend can stall on some cameras. This module
selects DirectShow unless config asks for the default backend.
"""

from __future__ import annotations

from typing import Any, Protocol

from perception.errors import CameraUnavailableError


class FrameCapture(Protocol):
    """The small part of ``cv2.VideoCapture`` the demo uses."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def set(self, prop: int, value: float) -> bool: ...


def backend_flag(backend: str, cv2_module: Any) -> int:
    """Map a config name to an OpenCV capture API flag."""

    if backend == "dshow":
        return int(cv2_module.CAP_DSHOW)
    if backend == "any":
        return int(cv2_module.CAP_ANY)
    raise CameraUnavailableError(
        f"Unknown camera backend {backend!r}. Use 'dshow' or 'any'."
    )


def open_camera(
    index: int,
    width: int,
    height: int,
    backend: str,
    *,
    cv2_module: Any,
    factory: Any | None = None,
) -> FrameCapture:
    """Open a camera and set the requested frame size.

    Raises ``CameraUnavailableError`` when the device cannot be opened.
    The capture is released before that exception is raised.
    """

    opener = factory or cv2_module.VideoCapture
    capture = opener(index, backend_flag(backend, cv2_module))
    if not capture.isOpened():
        capture.release()
        raise CameraUnavailableError(
            f"Could not open camera index {index} with backend {backend!r}. "
            "Check that the webcam is connected and not in use by another app."
        )
    capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def read_frame(capture: FrameCapture) -> tuple[bool, Any]:
    """Read one frame. A failed read is a boolean, not an exception."""

    ok, frame = capture.read()
    if not ok or frame is None:
        return False, None
    return True, frame
