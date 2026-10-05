"""Pilot gates: consent, confirmation, the frozen split, and pooled scores."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pilot import load_pilot, render_report, write_report  # noqa: E402
from evaluation.protocol import (  # noqa: E402
    CONDITIONS,
    confirm_trial,
    shift_raises,
    trial_document,
    write_trial_file,
)
from perception.errors import ConfigError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data" / "PROTOCOL.md"


def _raw(timestamp_ms: int, label: str) -> str:
    return json.dumps(
        {
            "record": "raw",
            "stream_timestamp_ms": timestamp_ms,
            "raw_label": label,
            "cue": "hand_raised",
        }
    )


def _start(timestamp_ms: int) -> str:
    return json.dumps(
        {"record": "filtered", "edge": "start", "stream_timestamp_ms": timestamp_ms, "cue": "hand_raised"}
    )


def _write_trial(
    root: Path,
    name: str,
    *,
    role: str,
    condition: str,
    followed_script: bool = True,
    participant_kind: str = "adult",
    consent_adult: bool = True,
    lines: list[str] | None = None,
    **overrides: object,
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    document = trial_document(
        trial_id=name,
        participant_code="P01",
        condition=CONDITIONS[condition],
        events=shift_raises(0, CONDITIONS[condition]),
        followed_script=followed_script,
    )
    document["role"] = role
    document["participant_kind"] = participant_kind
    document["consent_adult"] = consent_adult
    document.update(overrides)
    write_trial_file(directory / "trial.yaml", document)
    payload = lines if lines is not None else [_raw(0, "not_raised"), _raw(19000, "not_raised")]
    (directory / "predictions.jsonl").write_text("\n".join(payload) + "\n", encoding="utf-8")


class PilotTests(unittest.TestCase):
    def test_protocol_names_every_scheduled_condition(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("not a general validation", text)
        self.assertIn("1000 ms", text)
        for name in CONDITIONS:
            self.assertIn(name, text)

    def test_unconfirmed_tuning_and_non_adult_trials_stay_out(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_trial(
                root,
                "P01_positive",
                role="held_out",
                condition="positive",
                lines=[
                    _raw(0, "not_raised"),
                    _raw(1000, "raised"),
                    _raw(5000, "raised"),
                    _start(1300),
                    _raw(19000, "not_raised"),
                ],
                events=[{"start_ms": 1000, "end_ms": 4000}],
            )
            _write_trial(
                root,
                "P01_no_cue",
                role="held_out",
                condition="no_cue",
                lines=[_raw(0, "not_raised"), _raw(2000, "raised"), _raw(19000, "not_raised")],
            )
            _write_trial(root, "P01_positive_tuning", role="tuning", condition="positive_tuning")
            _write_trial(
                root,
                "P01_distance",
                role="held_out",
                condition="distance",
                followed_script=False,
            )
            _write_trial(
                root,
                "scripted",
                role="held_out",
                condition="lighting",
                participant_kind="constructed",
            )
            accepted, rejected = load_pilot(root)
            self.assertEqual(
                [trial.annotations.trial_id for trial in accepted if trial.role == "held_out"],
                ["P01_no_cue", "P01_positive"],
            )
            self.assertEqual([trial.annotations.trial_id for trial in accepted if trial.role == "tuning"], ["P01_positive_tuning"])
            reasons = {trial.trial_id: trial.reason for trial in rejected}
            self.assertIn("confirmed", reasons["P01_distance"])
            self.assertIn("adult", reasons["scripted"])
            report = render_report(accepted, rejected)
            self.assertIn("It is not a general validation.", report)
            self.assertIn("Held-out trials scored: 2", report)
            self.assertIn("Tuning trials kept out of the scores: 1", report)
            self.assertIn(
                "| pooled held-out | all scored | baseline | 0.3333 | 1.0000 | 0.5000 | 3.0 | 0.0 |",
                report,
            )
            self.assertIn(
                "| pooled held-out | all scored | filtered | 1.0000 | 1.0000 | 1.0000 | 0.0 | 300.0 |",
                report,
            )
            self.assertNotIn("P01_positive_tuning | positive_tuning | baseline", report)

    def test_empty_pilot_directory_claims_no_result(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "pilot.md"
            report = write_report(ROOT / "data" / "pilot", destination)
        self.assertIn("Held-out trials scored: 0", report)
        self.assertIn("Not measured.", report)
        self.assertIn("not a general validation", report)

    def test_confirm_requires_a_full_log(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "P01_positive"
            _write_trial(
                Path(tmp),
                "P01_positive",
                role="held_out",
                condition="positive",
                followed_script=False,
                lines=[_raw(0, "not_raised"), _raw(1000, "not_raised")],
            )
            with self.assertRaises(ConfigError):
                confirm_trial(root)
            lines = [_raw(1000, "not_raised"), _raw(21000, "not_raised")]
            (root / "predictions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(confirm_trial(root), "P01_positive")
            loaded = yaml.safe_load((root / "trial.yaml").read_text(encoding="utf-8"))
            self.assertIs(loaded["followed_script"], True)


if __name__ == "__main__":
    unittest.main()
