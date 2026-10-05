"""Score a hand-checked trial. This does not measure a live webcam session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.annotations import TrialAnnotations, load_annotations
from evaluation.log_reader import LogCounts, read_log
from evaluation.metrics import (
    EventScore,
    processing_rate_fps,
    score_events,
    unavailable_rate,
    unknown_rate,
)


def evaluate(annotations: TrialAnnotations, counts: LogCounts) -> dict[str, object]:
    starts = [event.start_ms for event in annotations.events]
    baseline = score_events(
        list(counts.baseline_ms),
        starts,
        tolerance_ms=annotations.tolerance_ms,
        duration_ms=annotations.duration_ms,
    )
    filtered = score_events(
        list(counts.filtered_ms),
        starts,
        tolerance_ms=annotations.tolerance_ms,
        duration_ms=annotations.duration_ms,
    )
    return {
        "trial_id": annotations.trial_id,
        "tolerance_ms": annotations.tolerance_ms,
        "duration_ms": annotations.duration_ms,
        "baseline": baseline,
        "filtered": filtered,
        "unknown_rate": unknown_rate(counts.unknown_frames, counts.raw_frames),
        "unavailable_rate": unavailable_rate(counts.unavailable_frames, counts.raw_frames),
        "processing_rate_fps": processing_rate_fps(counts.raw_frames, annotations.duration_ms),
    }


def evaluate_files(annotations_path: Path, log_path: Path) -> dict[str, object]:
    return evaluate(load_annotations(annotations_path), read_log(log_path))


def _format_delay(score: EventScore) -> str:
    if score.mean_delay_ms is None:
        return "n/a"
    return f"{score.mean_delay_ms:.1f}"


def _format_score(name: str, score: EventScore) -> str:
    return (
        f"{name:<10} "
        f"P {score.precision:.4f}  "
        f"R {score.recall:.4f}  "
        f"F1 {score.f1:.4f}  "
        f"FP/min {score.false_triggers_per_minute:.1f}  "
        f"delay_ms {_format_delay(score)}  "
        f"tp {score.true_positives} fp {score.false_positives} fn {score.false_negatives}"
    )


def format_report(report: dict[str, object]) -> str:
    baseline = report["baseline"]
    filtered = report["filtered"]
    assert isinstance(baseline, EventScore)
    assert isinstance(filtered, EventScore)
    lines = [
        f"trial {report['trial_id']}  tolerance_ms {report['tolerance_ms']}",
        "Hand-checked example. Not a pilot and not a live-session result.",
        _format_score("baseline", baseline),
        _format_score("filtered", filtered),
        f"unknown_rate {report['unknown_rate']:.4f}  "
        f"unavailable_rate {report['unavailable_rate']:.4f}  "
        f"processing_fps {report['processing_rate_fps']:.1f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a hand-checked MIRAGE-HRI trial")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/hand_checked/annotations.yaml"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/hand_checked/predictions.jsonl"),
    )
    args = parser.parse_args(argv)
    report = evaluate_files(args.annotations, args.log)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
