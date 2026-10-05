"""Score confirmed adult pilot trials. Tuning trials stay out of the table.

A trial is scored only when it is a consented adult, the script was confirmed,
no video was saved, the role is held out, and the frozen configuration matches.
This report is a pilot, not a general validation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.annotations import TrialAnnotations, load_annotations
from evaluation.log_reader import LogCounts, read_log
from evaluation.metrics import (
    EventScore,
    pool_scores,
    processing_rate_fps,
    score_events,
    unavailable_rate,
    unknown_rate,
)
from evaluation.protocol import (
    ACTIVATION_MS,
    COOLDOWN_MS,
    CONDITIONS,
    DURATION_MS,
    RAISE_MARGIN,
    RELEASE_MS,
    TOLERANCE_MS,
)
from perception.errors import ConfigError


@dataclass(frozen=True)
class PilotTrial:
    annotations: TrialAnnotations
    counts: LogCounts
    role: str
    condition: str
    participant_code: str


@dataclass(frozen=True)
class RejectedTrial:
    trial_id: str
    reason: str


def _reason(loaded: dict[str, object], directory: Path) -> str | None:
    if loaded.get("consent_adult") is not True:
        return "consent_adult is not true"
    if loaded.get("participant_kind") != "adult":
        return "participant_kind is not adult"
    if loaded.get("video_saved") is not False:
        return "video was saved"
    if loaded.get("followed_script") is not True:
        return "the participant has not confirmed the script"
    if loaded.get("cue") != "hand_raised":
        return "cue is not hand_raised"
    if loaded.get("role") not in {"held_out", "tuning"}:
        return "role is not held_out or tuning"
    if loaded.get("condition") not in CONDITIONS:
        return "condition is not in the pilot schedule"
    if int(loaded.get("duration_ms", 0)) != DURATION_MS:
        return "duration_ms is not the scheduled 20000"
    if int(loaded.get("tolerance_ms", -1)) != TOLERANCE_MS:
        return "tolerance_ms is not the frozen 1000"
    if int(loaded.get("activation_ms", -1)) != ACTIVATION_MS:
        return "activation_ms does not match the frozen filter"
    if int(loaded.get("release_ms", -1)) != RELEASE_MS:
        return "release_ms does not match the frozen filter"
    if int(loaded.get("cooldown_ms", -1)) != COOLDOWN_MS:
        return "cooldown_ms does not match the frozen filter"
    margin = loaded.get("raise_margin")
    if not isinstance(margin, (int, float)) or abs(float(margin) - RAISE_MARGIN) > 1e-9:
        return "raise_margin does not match the frozen rule"
    if not (directory / "predictions.jsonl").is_file():
        return "predictions.jsonl is missing"
    return None


def load_pilot(root: Path) -> tuple[tuple[PilotTrial, ...], tuple[RejectedTrial, ...]]:
    """Read trial folders. Folders without trial.yaml are ignored."""

    if not root.is_dir():
        raise ConfigError(f"Pilot directory not found: {root}")
    accepted: list[PilotTrial] = []
    rejected: list[RejectedTrial] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        document = directory / "trial.yaml"
        if not document.is_file():
            continue
        loaded = yaml.safe_load(document.read_text(encoding="utf-8"))
        trial_id = directory.name
        if isinstance(loaded, dict) and "trial_id" in loaded:
            trial_id = str(loaded["trial_id"])
        if not isinstance(loaded, dict):
            rejected.append(RejectedTrial(trial_id, "trial.yaml must contain a mapping"))
            continue
        reason = _reason(loaded, directory)
        if reason is not None:
            rejected.append(RejectedTrial(trial_id, reason))
            continue
        annotations = load_annotations(document)
        counts = read_log(directory / "predictions.jsonl")
        if counts.raw_frames == 0:
            rejected.append(RejectedTrial(trial_id, "the log has no raw frames"))
            continue
        accepted.append(
            PilotTrial(
                annotations=annotations,
                counts=counts,
                role=str(loaded["role"]),
                condition=str(loaded["condition"]),
                participant_code=str(loaded["participant_code"]),
            )
        )
    return tuple(accepted), tuple(rejected)


def _score(trial: PilotTrial) -> tuple[EventScore, EventScore]:
    starts = [event.start_ms for event in trial.annotations.events]
    baseline = score_events(
        list(trial.counts.baseline_ms),
        starts,
        tolerance_ms=trial.annotations.tolerance_ms,
        duration_ms=trial.annotations.duration_ms,
    )
    filtered = score_events(
        list(trial.counts.filtered_ms),
        starts,
        tolerance_ms=trial.annotations.tolerance_ms,
        duration_ms=trial.annotations.duration_ms,
    )
    return baseline, filtered


def _format_score(score: EventScore) -> str:
    delay = "n/a" if score.mean_delay_ms is None else f"{score.mean_delay_ms:.1f}"
    return (
        f"{score.precision:.4f} | {score.recall:.4f} | {score.f1:.4f} | "
        f"{score.false_triggers_per_minute:.1f} | {delay} | "
        f"{score.true_positives} | {score.false_positives} | {score.false_negatives}"
    )


def _condition_table() -> list[str]:
    lines = [
        "| Condition | Role | Instructed raises from the first frame |",
        "| --- | --- | --- |",
    ]
    for condition in CONDITIONS.values():
        if condition.raises:
            windows = ", ".join(f"{item.start_ms}–{item.end_ms} ms" for item in condition.raises)
        else:
            windows = "none"
        lines.append(f"| {condition.name} | {condition.role} | {windows} |")
    return lines


def render_report(
    accepted: tuple[PilotTrial, ...],
    rejected: tuple[RejectedTrial, ...],
) -> str:
    held_out = tuple(trial for trial in accepted if trial.role == "held_out")
    tuning = tuple(trial for trial in accepted if trial.role == "tuning")
    lines = [
        "# MIRAGE-HRI pilot report",
        "",
        "This is a pilot on a small set of consented adult trials. It is not a general validation.",
        "",
        "## Trial count",
        "",
        f"- Held-out trials scored: {len(held_out)}",
        f"- Tuning trials kept out of the scores: {len(tuning)}",
        f"- Trials present but not scored: {len(rejected)}",
        "",
    ]
    if tuning:
        lines.append(
            "Tuning trials: " + ", ".join(trial.annotations.trial_id for trial in tuning) + "."
        )
        lines.append("")
    if rejected:
        lines.append("Not scored:")
        lines.append("")
        for trial in rejected:
            lines.append(f"- {trial.trial_id}: {trial.reason}")
        lines.append("")
    lines.extend(
        [
            "## Conditions",
            "",
            "Positive raises, a no-cue period, transitions, and harder distance, lighting, and occlusion trials are all in the schedule. The tuning trial uses the same seated raise and stays out of the table.",
            "",
            *_condition_table(),
            "",
            "## Configuration",
            "",
            "These values were fixed before any pilot score. Baseline events are every raw `raised` frame. Filtered events are gesture starts. A match must fall within 1000 ms of the annotated start, and each annotation and each prediction is used once.",
            "",
            f"- tolerance_ms: {TOLERANCE_MS}",
            f"- activation_ms: {ACTIVATION_MS}",
            f"- release_ms: {RELEASE_MS}",
            f"- cooldown_ms: {COOLDOWN_MS}",
            f"- raise_margin: {RAISE_MARGIN}",
            f"- duration_ms: {DURATION_MS}",
            "",
            "## Results",
            "",
        ]
    )
    if not held_out:
        lines.extend(
            [
                "Not measured. No confirmed held-out adult trial is available, so no pilot result is claimed.",
                "",
            ]
        )
    else:
        scored = [(_score(trial), trial) for trial in held_out]
        baseline_pool = pool_scores(
            [pair[0][0] for pair in scored],
            [trial.annotations.duration_ms for _pair, trial in scored],
        )
        filtered_pool = pool_scores(
            [pair[0][1] for pair in scored],
            [trial.annotations.duration_ms for _pair, trial in scored],
        )
        raw_frames = sum(trial.counts.raw_frames for trial in held_out)
        unknown_frames = sum(trial.counts.unknown_frames for trial in held_out)
        unavailable_frames = sum(trial.counts.unavailable_frames for trial in held_out)
        duration_ms = sum(trial.annotations.duration_ms for trial in held_out)
        header = "| Trial | Condition | System | P | R | F1 | FP/min | delay_ms | TP | FP | FN |"
        rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        lines.extend([header, rule])
        for (baseline, filtered), trial in scored:
            trial_id = trial.annotations.trial_id
            lines.append(f"| {trial_id} | {trial.condition} | baseline | {_format_score(baseline)} |")
            lines.append(f"| {trial_id} | {trial.condition} | filtered | {_format_score(filtered)} |")
        lines.append(f"| pooled held-out | all scored | baseline | {_format_score(baseline_pool)} |")
        lines.append(f"| pooled held-out | all scored | filtered | {_format_score(filtered_pool)} |")
        lines.extend(
            [
                "",
                f"Held-out unknown rate {unknown_rate(unknown_frames, raw_frames):.4f}. "
                f"Unavailable rate {unavailable_rate(unavailable_frames, raw_frames):.4f}. "
                f"Processing rate {processing_rate_fps(raw_frames, duration_ms):.1f} frames per second "
                "of scheduled trial time.",
                "",
                "The useful comparison is false triggers against delay. A higher F1 is not assumed.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- The sample is a pilot. It does not validate the system for other people, rooms, or cameras.",
            "- Annotations are the instructed times, shifted to the first saved frame, and they count only after the participant confirms they followed that script.",
            "- No participant video is stored. Compliance is not checked with a second view.",
            "- `detector_score` is uncalibrated wrist-to-shoulder clearance, not a probability.",
            "- The only cue is `hand_raised`.",
            "- The hand-checked example under `data/hand_checked/` checks the scorer. It is not a pilot trial.",
            "- Live logs in `logs/` have no annotations and are not scored.",
            "- The tuning trial is reserved so a later threshold change can be tried without reusing the held-out trials. This report does not change the frozen settings.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(pilot_root: Path, destination: Path) -> str:
    accepted, rejected = load_pilot(pilot_root)
    report = render_report(accepted, rejected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the MIRAGE-HRI pilot report")
    parser.add_argument("--pilot", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output", type=Path, default=Path("reports/pilot.md"))
    args = parser.parse_args(argv)
    print(write_report(args.pilot, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
