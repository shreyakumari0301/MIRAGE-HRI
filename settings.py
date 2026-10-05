"""Load config.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from perception.errors import ConfigError


@dataclass(frozen=True)
class CameraSettings:
    index: int
    width: int
    height: int
    backend: str
    max_consecutive_read_failures: int


@dataclass(frozen=True)
class PoseSettings:
    model_path: Path
    num_poses: int
    min_pose_detection_confidence: float
    min_pose_presence_confidence: float
    min_tracking_confidence: float
    min_landmark_visibility: float
    min_visible_landmarks: int


@dataclass(frozen=True)
class AppSettings:
    camera: CameraSettings
    pose: PoseSettings
    window_name: str


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"config.yaml is missing the '{name}' section.")
    return section


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"config.yaml {section_name}.{key} is required.")
    return section[key]


def load_settings(path: Path) -> AppSettings:
    """Read settings. Relative model paths are resolved from the config file."""

    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping.")

    camera = _section(loaded, "camera")
    pose = _section(loaded, "pose")
    display = _section(loaded, "display")
    root = path.parent

    model_value = _required(pose, "model_path", "pose")
    model_path = Path(str(model_value))
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()

    backend = str(_required(camera, "backend", "camera"))
    if backend not in {"dshow", "any"}:
        raise ConfigError("camera.backend must be 'dshow' or 'any'.")

    try:
        return AppSettings(
            camera=CameraSettings(
                index=int(_required(camera, "index", "camera")),
                width=int(_required(camera, "width", "camera")),
                height=int(_required(camera, "height", "camera")),
                backend=backend,
                max_consecutive_read_failures=int(
                    _required(camera, "max_consecutive_read_failures", "camera")
                ),
            ),
            pose=PoseSettings(
                model_path=model_path,
                num_poses=int(_required(pose, "num_poses", "pose")),
                min_pose_detection_confidence=float(
                    _required(pose, "min_pose_detection_confidence", "pose")
                ),
                min_pose_presence_confidence=float(
                    _required(pose, "min_pose_presence_confidence", "pose")
                ),
                min_tracking_confidence=float(
                    _required(pose, "min_tracking_confidence", "pose")
                ),
                min_landmark_visibility=float(
                    _required(pose, "min_landmark_visibility", "pose")
                ),
                min_visible_landmarks=int(
                    _required(pose, "min_visible_landmarks", "pose")
                ),
            ),
            window_name=str(_required(display, "window_name", "display")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config.yaml has an invalid value: {exc}") from exc
