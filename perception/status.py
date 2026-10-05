"""Frame statuses shared by the pose demo.

These describe the landmark stream only. They are not interaction events
and they are not psychological labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FrameStatus(str, Enum):
    """What the pose step was able to say about one frame."""

    DETECTED = "detected"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Landmark:
    """One normalized landmark returned by the pose model.

    ``visibility`` is the model's own field, passed through unchanged.
    It is not a calibrated probability.
    """

    x: float
    y: float
    z: float
    visibility: float | None


@dataclass(frozen=True)
class PoseObservation:
    """Pose result for a single frame.

    ``landmarks`` is None when the model returned no pose, or when inference
    did not run. A partial pose can still be stored with status ``unknown``
    so the demo can show the points it actually received.
    """

    status: FrameStatus
    landmarks: tuple[Landmark, ...] | None
    visible_count: int
    message: str
    timestamp_ms: int
