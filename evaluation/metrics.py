"""Event-level scores.

A predicted event matches an annotated one when its emission time is within
``tolerance_ms`` of the annotated start. Each annotation and each prediction
are used at most once. Pairs are taken in order of closest time, so a second
prediction inside the same window is a duplicate false trigger.

Delay is predicted time minus annotated start. Positive means the event was late.
Precision is 0 when there are no predictions. Recall is 0 when there are no
annotations. F1 is 0 when precision and recall are both 0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchedPair:
    prediction_ms: int
    annotation_start_ms: int
    delay_ms: int


@dataclass(frozen=True)
class EventScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_triggers_per_minute: float
    mean_delay_ms: float | None
    matches: tuple[MatchedPair, ...]


def match_events(
    predicted_ms: list[int],
    annotated_starts_ms: list[int],
    tolerance_ms: int,
) -> tuple[tuple[MatchedPair, ...], int, int]:
    """Return matches, false-positive count, and false-negative count."""

    pairs: list[tuple[int, int, int, int]] = []
    for prediction_index, predicted in enumerate(predicted_ms):
        for annotation_index, start_ms in enumerate(annotated_starts_ms):
            distance = abs(predicted - start_ms)
            if distance <= tolerance_ms:
                pairs.append((distance, prediction_index, annotation_index, predicted - start_ms))
    pairs.sort()
    used_predictions: set[int] = set()
    used_annotations: set[int] = set()
    matches: list[MatchedPair] = []
    for _distance, prediction_index, annotation_index, delay_ms in pairs:
        if prediction_index in used_predictions or annotation_index in used_annotations:
            continue
        used_predictions.add(prediction_index)
        used_annotations.add(annotation_index)
        matches.append(
            MatchedPair(
                prediction_ms=predicted_ms[prediction_index],
                annotation_start_ms=annotated_starts_ms[annotation_index],
                delay_ms=delay_ms,
            )
        )
    false_positives = len(predicted_ms) - len(used_predictions)
    false_negatives = len(annotated_starts_ms) - len(used_annotations)
    return tuple(matches), false_positives, false_negatives


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_events(
    predicted_ms: list[int],
    annotated_starts_ms: list[int],
    *,
    tolerance_ms: int,
    duration_ms: int,
) -> EventScore:
    """Score one condition on one trial."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    matches, false_positives, false_negatives = match_events(
        predicted_ms, annotated_starts_ms, tolerance_ms
    )
    true_positives = len(matches)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    false_triggers_per_minute = false_positives / (duration_ms / 60_000)
    mean_delay = None if not matches else sum(pair.delay_ms for pair in matches) / len(matches)
    return EventScore(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        false_triggers_per_minute=false_triggers_per_minute,
        mean_delay_ms=mean_delay,
        matches=matches,
    )


def unknown_rate(unknown_frames: int, total_frames: int) -> float:
    return _ratio(unknown_frames, total_frames)


def unavailable_rate(unavailable_frames: int, total_frames: int) -> float:
    return _ratio(unavailable_frames, total_frames)


def processing_rate_fps(total_frames: int, duration_ms: int) -> float:
    """Frames divided by the trial duration. This is not a live benchmark."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    return total_frames / (duration_ms / 1000)
