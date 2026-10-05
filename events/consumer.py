"""Local stand-in for a robot that receives interaction events."""

from __future__ import annotations

from collections.abc import Callable

from events.schema import EventStatus, InteractionEvent


class RobotConsumer:
    """Accept events and respond only when the cue status is detected."""

    def __init__(self, writer: Callable[[str], None] | None = None) -> None:
        self.events: list[InteractionEvent] = []
        self.last_response: str | None = None
        self.overlay_line: str | None = None
        self._writer = print if writer is None else writer

    def receive(self, event: InteractionEvent) -> bool:
        """Return True when the robot acts on the event."""

        if event.status is not EventStatus.DETECTED or event.cue != "hand_raised":
            return False
        score = "null" if event.detector_score is None else f"{event.detector_score:.4f}"
        response = (
            f"robot: received {event.cue} "
            f"at {event.timestamp} "
            f"detector_score={score} ({event.detector_score_kind})"
        )
        self.events.append(event)
        self.last_response = response
        self.overlay_line = f"robot: hand_raised received  score={score} uncalibrated"
        self._writer(response)
        return True
