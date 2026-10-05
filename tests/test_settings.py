"""Config loading tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.errors import ConfigError  # noqa: E402
from settings import load_settings  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class SettingsTests(unittest.TestCase):
    def test_project_config_resolves_model_path(self) -> None:
        settings = load_settings(ROOT / "config.yaml")
        self.assertEqual(settings.camera.backend, "dshow")
        self.assertEqual(
            settings.pose.model_path,
            (ROOT / "models" / "pose_landmarker_lite.task").resolve(),
        )
        self.assertEqual(settings.pose.min_visible_landmarks, 8)
        self.assertEqual(settings.hand_raise.raise_margin, 0.08)
        self.assertEqual(settings.events.source, "webcam/hand_raise_rule")
        self.assertEqual(settings.events.log_path, (ROOT / "logs" / "session.jsonl").resolve())

    def test_bad_backend_is_a_config_error(self) -> None:
        text = (ROOT / "config.yaml").read_text(encoding="utf-8")
        text = text.replace("backend: dshow", "backend: msmf")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_settings(path)


if __name__ == "__main__":
    unittest.main()
