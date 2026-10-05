"""Trial annotations. Matching uses the annotated start, not the end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from perception.errors import ConfigError


@dataclass(frozen=True)
class AnnotatedEvent:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TrialAnnotations:
    trial_id: str
    duration_ms: int
    tolerance_ms: int
    cue: str
    events: tuple[AnnotatedEvent, ...]


def load_annotations(path: Path) -> TrialAnnotations:
    """Read one trial. ``tolerance_ms`` is the match window for this trial."""

    if not path.is_file():
        raise ConfigError(f"Annotations not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping.")
    try:
        events = tuple(
            AnnotatedEvent(start_ms=int(item["start_ms"]), end_ms=int(item["end_ms"]))
            for item in loaded["events"]
        )
        tolerance_ms = int(loaded["tolerance_ms"])
        duration_ms = int(loaded["duration_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Could not read annotations in {path}: {exc}") from exc
    if tolerance_ms < 0 or duration_ms <= 0:
        raise ConfigError("tolerance_ms must be >= 0 and duration_ms must be > 0.")
    return TrialAnnotations(
        trial_id=str(loaded["trial_id"]),
        duration_ms=duration_ms,
        tolerance_ms=tolerance_ms,
        cue=str(loaded.get("cue", "hand_raised")),
        events=events,
    )
