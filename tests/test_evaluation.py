"""Known answers for the event scorer, including duplicate predictions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.hand_checked import (  # noqa: E402
    ANNOTATIONS_PATH,
    LOG_PATH,
    build_log_lines,
)
from evaluation.metrics import score_events  # noqa: E402
from evaluation.run_evaluation import evaluate_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class MetricTests(unittest.TestCase):
    def test_duplicate_prediction_is_a_false_trigger(self) -> None:
        score = score_events(
            [1000, 1500],
            [1000],
            tolerance_ms=1000,
            duration_ms=60_000,
        )
        self.assertEqual(score.true_positives, 1)
        self.assertEqual(score.false_positives, 1)
        self.assertEqual(score.false_negatives, 0)
        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 1.0)
        self.assertAlmostEqual(score.f1, 2 / 3)
        self.assertEqual(score.matches[0].delay_ms, 0)
        self.assertEqual(score.false_triggers_per_minute, 1.0)

    def test_closer_annotation_wins_when_two_could_match(self) -> None:
        score = score_events([400], [0, 500], tolerance_ms=1000, duration_ms=10_000)
        self.assertEqual(score.true_positives, 1)
        self.assertEqual(score.false_negatives, 1)
        self.assertEqual(score.matches[0].annotation_start_ms, 500)
        self.assertEqual(score.matches[0].delay_ms, -100)

    def test_hand_checked_example_has_the_precomputed_scores(self) -> None:
        self.assertEqual(LOG_PATH.read_text(encoding="utf-8").splitlines(), build_log_lines())
        report = evaluate_files(ANNOTATIONS_PATH, LOG_PATH)
        self.assertEqual(report["tolerance_ms"], 1000)
        baseline = report["baseline"]
        filtered = report["filtered"]
        self.assertEqual(baseline.true_positives, 2)
        self.assertEqual(baseline.false_positives, 60)
        self.assertEqual(baseline.false_negatives, 0)
        self.assertAlmostEqual(baseline.precision, 2 / 62)
        self.assertEqual(baseline.recall, 1.0)
        self.assertAlmostEqual(baseline.f1, 0.0625)
        self.assertEqual(baseline.false_triggers_per_minute, 180.0)
        self.assertEqual(baseline.mean_delay_ms, 0.0)
        self.assertEqual(filtered.true_positives, 2)
        self.assertEqual(filtered.false_positives, 0)
        self.assertEqual(filtered.false_negatives, 0)
        self.assertEqual(filtered.precision, 1.0)
        self.assertEqual(filtered.recall, 1.0)
        self.assertEqual(filtered.f1, 1.0)
        self.assertEqual(filtered.false_triggers_per_minute, 0.0)
        self.assertEqual(filtered.mean_delay_ms, 300.0)
        self.assertEqual([pair.prediction_ms for pair in filtered.matches], [1300, 10300])
        self.assertAlmostEqual(report["unknown_rate"], 3 / 200)
        self.assertAlmostEqual(report["unavailable_rate"], 2 / 200)
        self.assertEqual(report["processing_rate_fps"], 10.0)


if __name__ == "__main__":
    unittest.main()
