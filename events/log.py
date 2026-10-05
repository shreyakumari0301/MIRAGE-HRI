"""Append-only JSONL log of raw labels and robot events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO


class JsonlLog:
    """Write one JSON object per line. Callers close the file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle: TextIO = path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, object]) -> None:
        self._handle.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
