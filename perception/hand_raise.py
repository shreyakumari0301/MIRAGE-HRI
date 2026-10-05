"""Raw frame-level hand-raise rule.

This is not an event detector. One call looks at one pose and returns one
label. Temporal filtering is a later phase.

Body sides follow the pose model, not the left and right edges of the image.
Index 15 is the person's left wrist and index 16 is the person's right wrist.
On an unmirrored webcam those appear swapped left-to-right.

Rule, for each side:

1. The wrist and shoulder must both be present.
2. Each visibility value must be finite and at least ``min_visibility``.
   Visibility is the model's own field, not a calibrated probability.
3. Both x and y must lie inside the frame, from 0 to 1 inclusive.
4. Image y grows downward. The side is raised when
   ``shoulder.y - wrist.y >= raise_margin``.

The frame label is ``raised`` when either side is raised, ``not_raised`` when
at least one side is usable and neither is raised, and ``insufficient`` when
neither side is usable. A missing point is left missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.status import Landmark

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

CUE_NAME = "hand_raised"


class HandRaiseLabel(str, Enum):
    """Raw cue label for one frame."""

    RAISED = "raised"
    NOT_RAISED = "not_raised"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class SideAssessment:
    """Wrist and shoulder evidence for one body side."""

    name: str
    wrist_index: int
    shoulder_index: int
    wrist_visibility: float | None
    shoulder_visibility: float | None
    clearance: float | None
    raised: bool

    @property
    def usable(self) -> bool:
        return self.clearance is not None


@dataclass(frozen=True)
class HandRaisePrediction:
    """One raw prediction. ``clearance`` is normalized image distance, not a probability."""

    label: HandRaiseLabel
    cue: str
    sides: tuple[SideAssessment, SideAssessment]
    message: str


def _at(
    landmarks: tuple[Landmark | None, ...] | None,
    index: int,
) -> Landmark | None:
    if landmarks is None or index >= len(landmarks):
        return None
    return landmarks[index]


def _usable(landmark: Landmark | None, min_visibility: float) -> bool:
    if landmark is None or landmark.visibility is None:
        return False
    if landmark.visibility < min_visibility:
        return False
    return 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0


def _assess_side(
    landmarks: tuple[Landmark | None, ...] | None,
    *,
    name: str,
    wrist_index: int,
    shoulder_index: int,
    min_visibility: float,
    raise_margin: float,
) -> SideAssessment:
    wrist = _at(landmarks, wrist_index)
    shoulder = _at(landmarks, shoulder_index)
    wrist_visibility = None if wrist is None else wrist.visibility
    shoulder_visibility = None if shoulder is None else shoulder.visibility
    if not _usable(wrist, min_visibility) or not _usable(shoulder, min_visibility):
        return SideAssessment(
            name=name,
            wrist_index=wrist_index,
            shoulder_index=shoulder_index,
            wrist_visibility=wrist_visibility,
            shoulder_visibility=shoulder_visibility,
            clearance=None,
            raised=False,
        )
    assert wrist is not None and shoulder is not None
    clearance = shoulder.y - wrist.y
    return SideAssessment(
        name=name,
        wrist_index=wrist_index,
        shoulder_index=shoulder_index,
        wrist_visibility=wrist_visibility,
        shoulder_visibility=shoulder_visibility,
        clearance=clearance,
        raised=clearance >= raise_margin,
    )


def _format_visibility(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def predict_hand_raise(
    landmarks: tuple[Landmark | None, ...] | None,
    *,
    min_visibility: float,
    raise_margin: float,
) -> HandRaisePrediction:
    """Apply the wrist/shoulder rule to one pose."""

    sides = (
        _assess_side(
            landmarks,
            name="left",
            wrist_index=LEFT_WRIST,
            shoulder_index=LEFT_SHOULDER,
            min_visibility=min_visibility,
            raise_margin=raise_margin,
        ),
        _assess_side(
            landmarks,
            name="right",
            wrist_index=RIGHT_WRIST,
            shoulder_index=RIGHT_SHOULDER,
            min_visibility=min_visibility,
            raise_margin=raise_margin,
        ),
    )
    raised = [side for side in sides if side.raised]
    if raised:
        label = HandRaiseLabel.RAISED
        names = ", ".join(side.name for side in raised)
        message = f"hand_raised: raised ({names})"
    elif any(side.usable for side in sides):
        label = HandRaiseLabel.NOT_RAISED
        message = "hand_raised: not_raised"
    else:
        label = HandRaiseLabel.INSUFFICIENT
        message = "hand_raised: insufficient landmarks"
    return HandRaisePrediction(label=label, cue=CUE_NAME, sides=sides, message=message)


def visibility_line(prediction: HandRaisePrediction) -> str:
    """Short overlay text: label plus wrist and shoulder visibility."""

    parts = [f"hand_raised: {prediction.label.value}"]
    for side in prediction.sides:
        parts.append(
            f"{side.name[0].upper()} w{_format_visibility(side.wrist_visibility)} "
            f"s{_format_visibility(side.shoulder_visibility)}"
        )
    return "  ".join(parts)
