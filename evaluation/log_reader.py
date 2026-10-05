"""Read a JSONL session into baseline times and filtered event times."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogCounts:
    raw_frames: int
    unknown_frames: int
    unavailable_frames: int
    baseline_ms: tuple[int, ...]
    filtered_ms: tuple[int, ...]


def read_log(path: Path) -> LogCounts:
    """Split raw raised frames from filtered gesture starts.

    A raw row with ``status`` ``unavailable`` counts as unavailable.
    Otherwise a raw label of ``insufficient`` counts as unknown.
    Baseline predictions are every raw ``raised`` frame. Filtered
    predictions are rows with ``record`` ``filtered`` and ``edge`` ``start``.
    """

    baseline: list[int] = []
    filtered: list[int] = []
    raw_frames = 0
    unknown_frames = 0
    unavailable_frames = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = row.get("record")
        if record == "raw":
            raw_frames += 1
            if row.get("status") == "unavailable":
                unavailable_frames += 1
            elif row.get("raw_label") == "insufficient":
                unknown_frames += 1
            if row.get("raw_label") == "raised":
                baseline.append(int(row["stream_timestamp_ms"]))
        elif record == "filtered" and row.get("edge") == "start":
            filtered.append(int(row["stream_timestamp_ms"]))
    return LogCounts(
        raw_frames=raw_frames,
        unknown_frames=unknown_frames,
        unavailable_frames=unavailable_frames,
        baseline_ms=tuple(baseline),
        filtered_ms=tuple(filtered),
    )


def stream_bounds_ms(path: Path) -> tuple[int, int] | None:
    """First and last raw stream timestamps, or None when the log has no raw rows."""

    first: int | None = None
    last: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record") != "raw" or "stream_timestamp_ms" not in row:
            continue
        timestamp_ms = int(row["stream_timestamp_ms"])
        if first is None:
            first = timestamp_ms
        last = timestamp_ms
    if first is None or last is None:
        return None
    return first, last
