"""Download the official MediaPipe Pose Landmarker lite model."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
MIN_BYTES = 1_000_000


def download(destination: Path = DESTINATION, url: str = MODEL_URL) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size >= MIN_BYTES:
        print(f"Model already present: {destination}")
        return destination
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    size = destination.stat().st_size
    if size < MIN_BYTES:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model is too small ({size} bytes).")
    print(f"Saved {destination} ({size} bytes)")
    return destination


if __name__ == "__main__":
    download()
