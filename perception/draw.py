"""Draw pose landmarks and the frame status onto a BGR image."""

from __future__ import annotations

from typing import Any

from perception.hand_raise import HandRaiseLabel, HandRaisePrediction, visibility_line
from perception.status import FrameStatus, Landmark, PoseObservation

# Standard 33-point MediaPipe pose topology, used only for display.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (27, 31),
    (28, 32),
    (29, 31),
    (30, 32),
)

_STATUS_COLOR = {
    FrameStatus.DETECTED: (60, 170, 60),
    FrameStatus.UNKNOWN: (0, 180, 220),
    FrameStatus.UNAVAILABLE: (50, 50, 220),
}


def _visible(landmark: Landmark, min_visibility: float) -> bool:
    return (
        landmark.visibility is not None
        and landmark.visibility >= min_visibility
        and 0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
    )


def _pixel(landmark: Landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


_HAND_COLOR = {
    HandRaiseLabel.RAISED: (60, 170, 60),
    HandRaiseLabel.NOT_RAISED: (220, 220, 220),
    HandRaiseLabel.INSUFFICIENT: (0, 180, 220),
}


def _mark_side(canvas: Any, points: dict[int, tuple[int, int]], prediction: HandRaisePrediction) -> None:
    import cv2

    for side in prediction.sides:
        if not side.raised:
            continue
        for index in (side.wrist_index, side.shoulder_index):
            if index in points:
                cv2.circle(canvas, points[index], 8, (60, 220, 60), 2, cv2.LINE_AA)


def render_frame(
    frame_bgr: Any,
    observation: PoseObservation,
    *,
    min_visibility: float,
    fps: float | None = None,
    hand_raise: HandRaisePrediction | None = None,
    robot_line: str | None = None,
) -> Any:
    """Draw usable landmarks and a status banner. The input frame is copied."""

    import cv2

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    landmarks = observation.landmarks or ()
    points: dict[int, tuple[int, int]] = {}
    for index, landmark in enumerate(landmarks):
        if landmark is not None and _visible(landmark, min_visibility):
            points[index] = _pixel(landmark, width, height)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], (255, 180, 40), 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(canvas, point, 3, (40, 220, 255), -1, cv2.LINE_AA)
    if hand_raise is not None:
        _mark_side(canvas, points, hand_raise)

    color = _STATUS_COLOR[observation.status]
    banner = f"{observation.status.value}: {observation.message}"
    if fps is not None:
        banner = f"{banner}  fps {fps:.1f}"
    lines = [banner]
    line_colors = [color]
    if hand_raise is not None:
        lines.append(visibility_line(hand_raise))
        line_colors.append(_HAND_COLOR[hand_raise.label])
    if robot_line:
        lines.append(robot_line)
        line_colors.append((220, 220, 220))
    banner_height = 22 + 18 * len(lines)
    top = max(height - banner_height, 0)
    cv2.rectangle(canvas, (0, top), (width, height), (20, 20, 20), -1)
    for offset, (text, text_color) in enumerate(zip(lines, line_colors)):
        baseline = min(top + 18 + 18 * offset, height - 4)
        cv2.putText(
            canvas,
            text,
            (8, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return canvas
