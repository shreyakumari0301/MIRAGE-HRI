"""Pilot schedule. The split and the filter were fixed before any pilot score.

``positive_tuning`` is recorded with the same settings and is not scored.
The other conditions are held out. Changing ``config.yaml`` after a trial
does not belong in this report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from evaluation.annotations import AnnotatedEvent
from evaluation.log_reader import stream_bounds_ms
from perception.errors import ConfigError

TOLERANCE_MS = 1000
DURATION_MS = 20_000
ACTIVATION_MS = 300
RELEASE_MS = 400
COOLDOWN_MS = 500
RAISE_MARGIN = 0.08
PARTICIPANT_CODE = re.compile(r"^P[0-9]{2,}$")


@dataclass(frozen=True)
class ScheduledRaise:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Condition:
    name: str
    role: str
    instruction: str
    raises: tuple[ScheduledRaise, ...]


CONDITIONS: dict[str, Condition] = {
    "positive": Condition(
        "positive",
        "held_out",
        "Seated, normal light, about an arm's length from the camera.",
        (ScheduledRaise(3000, 6000), ScheduledRaise(12000, 15000)),
    ),
    "no_cue": Condition(
        "no_cue",
        "held_out",
        "Same seat and light. Keep both hands down for the whole trial.",
        (),
    ),
    "transitions": Condition(
        "transitions",
        "held_out",
        "Same seat and light. Raise, lower, and raise again on the RAISE cue.",
        (ScheduledRaise(2000, 4000), ScheduledRaise(7000, 9000), ScheduledRaise(13000, 16000)),
    ),
    "distance": Condition(
        "distance",
        "held_out",
        "Step back about two meters. Raise on the RAISE cue.",
        (ScheduledRaise(4000, 7000),),
    ),
    "lighting": Condition(
        "lighting",
        "held_out",
        "Dim the room and stay in frame. Raise on the RAISE cue.",
        (ScheduledRaise(4000, 7000),),
    ),
    "occlusion": Condition(
        "occlusion",
        "held_out",
        "Hold a folder so one shoulder is partly hidden. Raise on the RAISE cue.",
        (ScheduledRaise(4000, 7000),),
    ),
    "positive_tuning": Condition(
        "positive_tuning",
        "tuning",
        "Same seated raise as the positive trial. This trial is not scored.",
        (ScheduledRaise(3000, 6000),),
    ),
}


def condition_named(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        known = ", ".join(CONDITIONS)
        raise ConfigError(f"Unknown condition {name!r}. Known conditions: {known}.") from exc


def check_participant_code(code: str) -> str:
    if PARTICIPANT_CODE.fullmatch(code) is None:
        raise ConfigError("Use a participant code such as P01. Do not store a name.")
    return code


def shift_raises(first_stream_ms: int, condition: Condition) -> tuple[AnnotatedEvent, ...]:
    """Place instructed raises on the stream clock of the first saved frame."""

    return tuple(
        AnnotatedEvent(first_stream_ms + item.start_ms, first_stream_ms + item.end_ms)
        for item in condition.raises
    )


def prompt_at(condition: Condition, elapsed_ms: int) -> str:
    """Short line for the webcam window. RAISE means the hand should be up."""

    if any(item.start_ms <= elapsed_ms < item.end_ms for item in condition.raises):
        return f"RAISE and hold. {condition.name}"
    if condition.name == "no_cue":
        return "Hands down. no_cue"
    return f"Hands down. {condition.name}"


def trial_document(
    *,
    trial_id: str,
    participant_code: str,
    condition: Condition,
    events: tuple[AnnotatedEvent, ...],
    followed_script: bool,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "participant_code": participant_code,
        "consent_adult": True,
        "participant_kind": "adult",
        "video_saved": False,
        "followed_script": followed_script,
        "role": condition.role,
        "condition": condition.name,
        "duration_ms": DURATION_MS,
        "tolerance_ms": TOLERANCE_MS,
        "cue": "hand_raised",
        "activation_ms": ACTIVATION_MS,
        "release_ms": RELEASE_MS,
        "cooldown_ms": COOLDOWN_MS,
        "raise_margin": RAISE_MARGIN,
        "annotation_origin": "instructed_from_first_frame",
        "events": [{"start_ms": event.start_ms, "end_ms": event.end_ms} for event in events],
    }


def write_trial_file(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _matches_frozen(loaded: dict[str, object]) -> bool:
    try:
        margin = float(loaded["raise_margin"])  # type: ignore[arg-type]
        return (
            int(loaded["activation_ms"]) == ACTIVATION_MS  # type: ignore[arg-type]
            and int(loaded["release_ms"]) == RELEASE_MS  # type: ignore[arg-type]
            and int(loaded["cooldown_ms"]) == COOLDOWN_MS  # type: ignore[arg-type]
            and int(loaded["tolerance_ms"]) == TOLERANCE_MS  # type: ignore[arg-type]
            and math.isclose(margin, RAISE_MARGIN)
        )
    except (KeyError, TypeError, ValueError):
        return False


def confirm_trial(trial_dir: Path) -> str:
    """Mark a recorded trial as script-following when the log covers the full trial."""

    path = trial_dir / "trial.yaml"
    log_path = trial_dir / "predictions.jsonl"
    if not path.is_file() or not log_path.is_file():
        raise ConfigError(f"{trial_dir} needs trial.yaml and predictions.jsonl.")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping.")
    if loaded.get("consent_adult") is not True or loaded.get("participant_kind") != "adult":
        raise ConfigError("Only a consented adult trial can be confirmed.")
    if loaded.get("video_saved") is not False:
        raise ConfigError("A pilot trial must not have saved video.")
    if not _matches_frozen(loaded):
        raise ConfigError("This trial does not use the frozen pilot configuration.")
    bounds = stream_bounds_ms(log_path)
    if bounds is None:
        raise ConfigError("The log has no raw frames.")
    first_ms, last_ms = bounds
    if last_ms - first_ms < DURATION_MS - TOLERANCE_MS:
        raise ConfigError(
            "The recording is shorter than the 20 second script, so it stays unconfirmed."
        )
    loaded["followed_script"] = True
    write_trial_file(path, loaded)
    return str(loaded.get("trial_id", trial_dir.name))
