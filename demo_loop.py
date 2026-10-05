"""Webcam loop for the Phase 1 landmark demo.

The loop counts statuses and keeps running through unknown frames and short
camera dropouts. It stops cleanly if the camera stays unreadable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from perception.camera import FrameCapture, read_frame
from perception.errors import CameraUnavailableError
from perception.status import FrameStatus, PoseObservation


DetectFn = Callable[[Any], PoseObservation]
RenderFn = Callable[[Any, PoseObservation, float | None], Any]
ShowFn = Callable[[Any], None]


@dataclass
class LoopSummary:
    """Counts from one demo run. These are not an evaluation."""

    exit_code: int
    frames: int = 0
    detected: int = 0
    unknown: int = 0
    unavailable: int = 0
    elapsed_s: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.frames / self.elapsed_s


def run_demo(
    capture: FrameCapture,
    detect: DetectFn,
    render: RenderFn,
    *,
    max_frames: int | None,
    max_consecutive_read_failures: int,
    show: ShowFn | None = None,
    wait_key: Callable[[], int] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> LoopSummary:
    """Process frames until the user quits, the frame cap is hit, or the camera dies."""

    summary = LoopSummary(exit_code=0)
    consecutive_failures = 0
    smoothed_fps: float | None = None
    previous = clock()
    last_status: FrameStatus | None = None
    started = clock()

    try:
        while max_frames is None or summary.frames < max_frames:
            ok, frame = read_frame(capture)
            now = clock()
            dt = now - previous
            previous = now
            if dt > 0:
                instant = 1.0 / dt
                smoothed_fps = instant if smoothed_fps is None else (0.9 * smoothed_fps + 0.1 * instant)

            if not ok:
                consecutive_failures += 1
                summary.unavailable += 1
                message = "Camera read failed."
                if last_status is not FrameStatus.UNAVAILABLE:
                    summary.messages.append(message)
                    last_status = FrameStatus.UNAVAILABLE
                if consecutive_failures >= max_consecutive_read_failures:
                    summary.exit_code = 1
                    summary.messages.append(
                        "Camera stayed unreadable. "
                        f"Stopped after {consecutive_failures} failed reads."
                    )
                    break
                continue

            consecutive_failures = 0
            observation = detect(frame)
            summary.frames += 1
            if observation.status is FrameStatus.DETECTED:
                summary.detected += 1
            elif observation.status is FrameStatus.UNKNOWN:
                summary.unknown += 1
            else:
                summary.unavailable += 1
            if observation.status is not last_status:
                summary.messages.append(observation.message)
                last_status = observation.status

            image = render(frame, observation, smoothed_fps)
            if show is not None:
                show(image)
            if wait_key is not None and wait_key() in {ord("q"), 27}:
                summary.messages.append("Stopped.")
                break
    finally:
        summary.elapsed_s = clock() - started
    return summary


def ensure_live_camera(capture: FrameCapture, index: int) -> None:
    """Fail before the loop when the capture never opened."""

    if not capture.isOpened():
        raise CameraUnavailableError(
            f"Camera index {index} is not open. No landmarks will be invented."
        )
