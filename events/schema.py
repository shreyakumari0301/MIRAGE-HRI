"""Interaction events sent to the simulated robot.

``detector_score`` is the wrist-to-shoulder clearance in normalized image
coordinates. The field ``detector_score_kind`` is ``uncalibrated_clearance``.
That number is not a probability and it is not a confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from perception.hand_raise import HandRaiseLabel, HandRaisePrediction
from perception.status import FrameStatus, PoseObservation

SCHEMA_VERSION = 1
DETECTOR_SCORE_KIND = "uncalibrated_clearance"
CUE_NAME = "hand_raised"

Clock = Callable[[], datetime]


class EventStatus(str, Enum):
    """What the robot-facing event is allowed to claim."""

    DETECTED = "detected"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


def format_timestamp(moment: datetime) -> str:
    """UTC timestamp with millisecond precision."""

    utc = moment.astimezone(timezone.utc)
    millis = utc.microsecond // 1000
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"


def _round_score(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def detector_score(prediction: HandRaisePrediction) -> float | None:
    """Clearance for a raised hand, or the largest usable clearance otherwise.

    Returns None when neither wrist/shoulder pair was usable. Missing
    measurements stay missing.
    """

    raised = [
        side.clearance
        for side in prediction.sides
        if side.raised and side.clearance is not None
    ]
    if raised:
        return max(raised)
    usable = [side.clearance for side in prediction.sides if side.clearance is not None]
    if not usable:
        return None
    return max(usable)


@dataclass(frozen=True)
class RawFrameRecord:
    """One raw hand-raise label, kept separate from robot events."""

    timestamp: str
    stream_timestamp_ms: int
    cue: str
    raw_label: str
    detector_score: float | None
    detector_score_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "record": "raw",
            "timestamp": self.timestamp,
            "stream_timestamp_ms": self.stream_timestamp_ms,
            "cue": self.cue,
            "raw_label": self.raw_label,
            "detector_score": self.detector_score,
            "detector_score_kind": self.detector_score_kind,
        }


@dataclass(frozen=True)
class InteractionEvent:
    """One timestamped event. Status is detected, unknown, or unavailable."""

    schema_version: int
    timestamp: str
    stream_timestamp_ms: int
    cue: str
    status: EventStatus
    raw_label: str
    detector_score: float | None
    detector_score_kind: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "record": "event",
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "stream_timestamp_ms": self.stream_timestamp_ms,
            "cue": self.cue,
            "status": self.status.value,
            "raw_label": self.raw_label,
            "detector_score": self.detector_score,
            "detector_score_kind": self.detector_score_kind,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))


def raw_frame_record(
    observation: PoseObservation,
    prediction: HandRaisePrediction,
    *,
    clock: Clock,
) -> RawFrameRecord:
    """Copy the raw label. This does not decide whether the robot should act."""

    return RawFrameRecord(
        timestamp=format_timestamp(clock()),
        stream_timestamp_ms=observation.timestamp_ms,
        cue=prediction.cue,
        raw_label=prediction.label.value,
        detector_score=_round_score(detector_score(prediction)),
        detector_score_kind=DETECTOR_SCORE_KIND,
    )


def build_interaction_event(
    observation: PoseObservation,
    prediction: HandRaisePrediction,
    *,
    source: str,
    clock: Clock,
) -> InteractionEvent | None:
    """Map one frame onto an event.

    ``raised`` becomes status ``detected``. ``insufficient`` becomes
    ``unknown``. A failed pose step becomes ``unavailable`` and does not
    invent a raised hand. ``not_raised`` is a completed look with the cue
    absent, so it stays in the raw log and does not become an event.
    """

    if observation.status is FrameStatus.UNAVAILABLE:
        status = EventStatus.UNAVAILABLE
        score = None
    elif prediction.label is HandRaiseLabel.RAISED:
        status = EventStatus.DETECTED
        score = _round_score(detector_score(prediction))
    elif prediction.label is HandRaiseLabel.INSUFFICIENT:
        status = EventStatus.UNKNOWN
        score = None
    else:
        return None

    return InteractionEvent(
        schema_version=SCHEMA_VERSION,
        timestamp=format_timestamp(clock()),
        stream_timestamp_ms=observation.timestamp_ms,
        cue=CUE_NAME,
        status=status,
        raw_label=prediction.label.value,
        detector_score=score,
        detector_score_kind=DETECTOR_SCORE_KIND,
        source=source,
    )
