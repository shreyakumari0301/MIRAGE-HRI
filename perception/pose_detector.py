"""MediaPipe Pose wrapper.

Frame-level landmark results stay in this module. The hand-raise rule reads
those landmarks from ``perception.hand_raise`` and does not live here.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Protocol

from perception.errors import PoseModelError
from perception.status import FrameStatus, Landmark, PoseObservation


class _Landmarker(Protocol):
    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...

    def close(self) -> None: ...


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def landmark_from_raw(raw: Any) -> Landmark | None:
    """Copy one model landmark. Non-finite coordinates are rejected."""

    x = _as_float(getattr(raw, "x", None))
    y = _as_float(getattr(raw, "y", None))
    z = _as_float(getattr(raw, "z", None))
    if x is None or y is None or z is None:
        return None
    visibility = _as_float(getattr(raw, "visibility", None))
    return Landmark(x=x, y=y, z=z, visibility=visibility)


def observation_from_landmarks(
    raw_landmarks: list[Any] | None,
    timestamp_ms: int,
    *,
    min_visibility: float,
    min_visible_landmarks: int,
) -> PoseObservation:
    """Turn one pose's raw landmarks into a status.

    ``detected`` requires at least ``min_visible_landmarks`` points whose
    model visibility is finite and at least ``min_visibility``. Fewer visible
    points, or no pose at all, is ``unknown``. A rejected point stays an empty
    slot so later indexes still match the pose model. Missing points are not
    replaced with guessed coordinates.
    """

    if not raw_landmarks:
        return PoseObservation(
            status=FrameStatus.UNKNOWN,
            landmarks=None,
            visible_count=0,
            message="No pose landmarks in frame.",
            timestamp_ms=timestamp_ms,
        )

    converted: list[Landmark | None] = []
    visible_count = 0
    for raw in raw_landmarks:
        landmark = landmark_from_raw(raw)
        converted.append(landmark)
        if (
            landmark is not None
            and landmark.visibility is not None
            and landmark.visibility >= min_visibility
        ):
            visible_count += 1

    landmarks = tuple(converted)
    if visible_count < min_visible_landmarks:
        return PoseObservation(
            status=FrameStatus.UNKNOWN,
            landmarks=landmarks or None,
            visible_count=visible_count,
            message=(
                "Insufficient landmark visibility "
                f"({visible_count}/{min_visible_landmarks})."
            ),
            timestamp_ms=timestamp_ms,
        )
    return PoseObservation(
        status=FrameStatus.DETECTED,
        landmarks=landmarks,
        visible_count=visible_count,
        message=f"Pose landmarks visible ({visible_count}).",
        timestamp_ms=timestamp_ms,
    )


def observation_from_result(
    result: Any,
    timestamp_ms: int,
    *,
    min_visibility: float,
    min_visible_landmarks: int,
) -> PoseObservation:
    """Read the first pose from a MediaPipe landmarker result."""

    if result is None or not hasattr(result, "pose_landmarks"):
        return PoseObservation(
            status=FrameStatus.UNAVAILABLE,
            landmarks=None,
            visible_count=0,
            message="Pose model returned no usable result.",
            timestamp_ms=timestamp_ms,
        )
    poses = result.pose_landmarks or []
    if not poses:
        return observation_from_landmarks(
            None,
            timestamp_ms,
            min_visibility=min_visibility,
            min_visible_landmarks=min_visible_landmarks,
        )
    return observation_from_landmarks(
        list(poses[0]),
        timestamp_ms,
        min_visibility=min_visibility,
        min_visible_landmarks=min_visible_landmarks,
    )


def load_landmarker(model_path: Path, pose_options: dict[str, float | int]) -> Any:
    """Create a VIDEO-mode PoseLandmarker, or raise ``PoseModelError``."""

    if not model_path.is_file():
        raise PoseModelError(
            f"Pose model not found at {model_path}. "
            "Run: .\\.venv\\Scripts\\python.exe scripts\\download_pose_model.py"
        )
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise PoseModelError(
            "MediaPipe is not installed in this interpreter. "
            "Use the project virtual environment."
        ) from exc

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=int(pose_options["num_poses"]),
        min_pose_detection_confidence=float(
            pose_options["min_pose_detection_confidence"]
        ),
        min_pose_presence_confidence=float(
            pose_options["min_pose_presence_confidence"]
        ),
        min_tracking_confidence=float(pose_options["min_tracking_confidence"]),
    )
    try:
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        raise PoseModelError(f"Could not load the pose model: {exc}") from exc


class PoseDetector:
    """Run pose landmarking on BGR frames from OpenCV.

    A bad frame or an inference error becomes status ``unavailable``.
    It does not raise into the webcam loop and it does not invent landmarks.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_landmark_visibility: float = 0.5,
        min_visible_landmarks: int = 8,
        landmarker: _Landmarker | None = None,
        image_factory: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.min_landmark_visibility = min_landmark_visibility
        self.min_visible_landmarks = min_visible_landmarks
        self._image_factory = image_factory
        self._clock = time.perf_counter
        if landmarker is None:
            self._landmarker = load_landmarker(
                model_path,
                {
                    "num_poses": num_poses,
                    "min_pose_detection_confidence": min_pose_detection_confidence,
                    "min_pose_presence_confidence": min_pose_presence_confidence,
                    "min_tracking_confidence": min_tracking_confidence,
                },
            )
        else:
            self._landmarker = landmarker
        self._t0 = self._clock()
        self._timestamp_ms = 0
        self._closed = False

    def next_timestamp_ms(self) -> int:
        """Milliseconds since construction, strictly increasing."""

        now = int((self._clock() - self._t0) * 1000)
        if now <= self._timestamp_ms:
            now = self._timestamp_ms + 1
        self._timestamp_ms = now
        return now

    def detect(self, frame_bgr: Any) -> PoseObservation:
        timestamp_ms = self.next_timestamp_ms()
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            return PoseObservation(
                status=FrameStatus.UNAVAILABLE,
                landmarks=None,
                visible_count=0,
                message="Camera frame is missing or not a color image.",
                timestamp_ms=timestamp_ms,
            )
        if frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
            return PoseObservation(
                status=FrameStatus.UNAVAILABLE,
                landmarks=None,
                visible_count=0,
                message="Camera frame is missing or not a color image.",
                timestamp_ms=timestamp_ms,
            )
        try:
            image = self._to_mp_image(frame_bgr)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            return PoseObservation(
                status=FrameStatus.UNAVAILABLE,
                landmarks=None,
                visible_count=0,
                message=f"Pose inference failed: {exc}",
                timestamp_ms=timestamp_ms,
            )
        return observation_from_result(
            result,
            timestamp_ms,
            min_visibility=self.min_landmark_visibility,
            min_visible_landmarks=self.min_visible_landmarks,
        )

    def _to_mp_image(self, frame_bgr: Any) -> Any:
        if self._image_factory is not None:
            return self._image_factory(frame_bgr)
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._landmarker, "close", None)
        if closer is not None:
            closer()
