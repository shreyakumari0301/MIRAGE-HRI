"""Build the small example whose scores are known before the file is read.

The trial is 20 seconds at 10 frames per second. Two annotated raises are
held. Two extra raw raised frames are one frame long, so the 300 ms filter
does not emit them. Three frames are insufficient and two are unavailable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from events.temporal_filter import TemporalFilter
from perception.hand_raise import HandRaiseLabel, HandRaisePrediction, SideAssessment
from perception.status import FrameStatus, PoseObservation

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "data" / "hand_checked" / "annotations.yaml"
LOG_PATH = ROOT / "data" / "hand_checked" / "predictions.jsonl"

TOLERANCE_MS = 1000
DURATION_MS = 20_000
ACTIVATION_MS = 300
RELEASE_MS = 400
COOLDOWN_MS = 500


def frame_label(timestamp_ms: int) -> tuple[str, str | None]:
    """Return the raw label and an optional unavailable status."""

    if timestamp_ms in {0, 100, 200}:
        return "insufficient", None
    if timestamp_ms in {300, 400}:
        return "not_raised", "unavailable"
    if 1000 <= timestamp_ms <= 3900 or 10000 <= timestamp_ms <= 12900:
        return "raised", None
    if timestamp_ms in {5000, 15000}:
        return "raised", None
    return "not_raised", None


def _prediction(label: str) -> HandRaisePrediction:
    raised = label == "raised"
    kind = {
        "raised": HandRaiseLabel.RAISED,
        "insufficient": HandRaiseLabel.INSUFFICIENT,
    }.get(label, HandRaiseLabel.NOT_RAISED)
    side = SideAssessment(
        name="right",
        wrist_index=16,
        shoulder_index=12,
        wrist_visibility=0.9,
        shoulder_visibility=0.9,
        clearance=0.25 if raised else None,
        raised=raised,
    )
    other = SideAssessment("left", 15, 11, 0.2, 0.2, None, False)
    return HandRaisePrediction(label=kind, cue="hand_raised", sides=(other, side), message=kind.value)


def build_log_lines() -> list[str]:
    """Raw rows plus the filtered start and end rows from the real filter."""

    gesture = TemporalFilter(ACTIVATION_MS, RELEASE_MS, COOLDOWN_MS)
    clock = lambda: datetime(2026, 10, 5, tzinfo=timezone.utc)
    lines: list[str] = []
    for timestamp_ms in range(0, DURATION_MS, 100):
        label, status = frame_label(timestamp_ms)
        raw = {
            "record": "raw",
            "timestamp": "2026-10-05T00:00:00.000Z",
            "stream_timestamp_ms": timestamp_ms,
            "cue": "hand_raised",
            "raw_label": label,
            "detector_score": 0.25 if label == "raised" else None,
            "detector_score_kind": "uncalibrated_clearance",
        }
        if status is not None:
            raw["status"] = status
        lines.append(json.dumps(raw, separators=(",", ":")))
        observation = PoseObservation(
            status=FrameStatus.UNAVAILABLE if status == "unavailable" else FrameStatus.DETECTED,
            landmarks=None,
            visible_count=0,
            message="example",
            timestamp_ms=timestamp_ms,
        )
        update = gesture.update(
            observation,
            _prediction(label),
            source="hand_checked",
            clock=clock,
        )
        logged = update.to_log()
        if logged is not None:
            lines.append(json.dumps(logged, separators=(",", ":")))
    return lines


def write_example(log_path: Path = LOG_PATH) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(build_log_lines()) + "\n", encoding="utf-8")
    return log_path
