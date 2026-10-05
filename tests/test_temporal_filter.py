"""Flicker goes in. One gesture event comes out. No camera is required."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from events.consumer import RobotConsumer  # noqa: E402
from events.temporal_filter import TemporalFilter  # noqa: E402
from perception.hand_raise import (  # noqa: E402
    HandRaiseLabel,
    HandRaisePrediction,
    SideAssessment,
)
from perception.status import FrameStatus, PoseObservation  # noqa: E402

MOMENT = datetime(2026, 10, 5, 13, 0, 0, tzinfo=timezone.utc)


def _clock() -> datetime:
    return MOMENT


def _prediction(label: HandRaiseLabel) -> HandRaisePrediction:
    raised = label is HandRaiseLabel.RAISED
    right = SideAssessment(
        name="right",
        wrist_index=16,
        shoulder_index=12,
        wrist_visibility=0.9,
        shoulder_visibility=0.9,
        clearance=0.3 if raised else None,
        raised=raised,
    )
    left = SideAssessment("left", 15, 11, 0.2, 0.2, None, False)
    return HandRaisePrediction(label=label, cue="hand_raised", sides=(left, right), message=label.value)


def _observation(timestamp_ms: int, status: FrameStatus = FrameStatus.DETECTED) -> PoseObservation:
    return PoseObservation(
        status=status,
        landmarks=None,
        visible_count=0,
        message="test",
        timestamp_ms=timestamp_ms,
    )


def _play(
    samples: list[tuple[int, HandRaiseLabel]],
    *,
    activation_ms: int = 200,
    release_ms: int = 400,
    cooldown_ms: int = 500,
    status: FrameStatus = FrameStatus.DETECTED,
) -> list:
    gesture = TemporalFilter(activation_ms, release_ms, cooldown_ms)
    emitted = []
    for timestamp_ms, label in samples:
        update = gesture.update(
            _observation(timestamp_ms, status),
            _prediction(label),
            source="test",
            clock=_clock,
        )
        if update.event is not None:
            emitted.append(update)
    return emitted


class TemporalFilterTests(unittest.TestCase):
    def test_flicker_then_a_hold_emits_one_event_after_the_wait(self) -> None:
        samples = [
            (0, HandRaiseLabel.RAISED),
            (40, HandRaiseLabel.NOT_RAISED),
            (80, HandRaiseLabel.RAISED),
            (120, HandRaiseLabel.INSUFFICIENT),
            (200, HandRaiseLabel.RAISED),
            (280, HandRaiseLabel.RAISED),
            (360, HandRaiseLabel.RAISED),
            (400, HandRaiseLabel.RAISED),
            (480, HandRaiseLabel.RAISED),
            (800, HandRaiseLabel.NOT_RAISED),
            (1200, HandRaiseLabel.NOT_RAISED),
        ]
        emitted = _play(samples)
        self.assertEqual(len(emitted), 1)
        start = emitted[0]
        self.assertEqual(start.edge, "start")
        self.assertEqual(start.delay_ms, 200)
        assert start.event is not None
        self.assertEqual(start.event.stream_timestamp_ms, 400)
        self.assertEqual(start.event.status.value, "detected")
        self.assertEqual(start.event.timestamp, "2026-10-05T13:00:00.000Z")
        logged = start.to_log()
        assert logged is not None
        self.assertEqual(logged["record"], "filtered")
        self.assertEqual(logged["delay_ms"], 200)
        self.assertEqual(logged["raw_label"], "raised")

        lines: list[str] = []
        consumer = RobotConsumer(writer=lines.append)
        self.assertTrue(consumer.receive(start.event))
        self.assertEqual(len(consumer.events), 1)
        self.assertEqual(len(lines), 1)

    def test_a_single_raised_frame_emits_nothing(self) -> None:
        emitted = _play(
            [
                (0, HandRaiseLabel.RAISED),
                (40, HandRaiseLabel.NOT_RAISED),
                (80, HandRaiseLabel.INSUFFICIENT),
            ]
        )
        self.assertEqual(emitted, [])

    def test_a_second_hold_inside_cooldown_is_ignored(self) -> None:
        first = [(0, HandRaiseLabel.RAISED), (200, HandRaiseLabel.RAISED)]
        release = [(200, HandRaiseLabel.NOT_RAISED), (600, HandRaiseLabel.NOT_RAISED)]
        during_cooldown = [
            (700, HandRaiseLabel.RAISED),
            (900, HandRaiseLabel.RAISED),
            (1100, HandRaiseLabel.RAISED),
        ]
        emitted = _play(first + release + during_cooldown, release_ms=400, cooldown_ms=800)
        self.assertEqual(len(emitted), 1)

    def test_a_second_hold_after_cooldown_emits_again(self) -> None:
        first = [(0, HandRaiseLabel.RAISED), (200, HandRaiseLabel.RAISED)]
        release = [(200, HandRaiseLabel.NOT_RAISED), (600, HandRaiseLabel.NOT_RAISED)]
        second = [(1200, HandRaiseLabel.RAISED), (1400, HandRaiseLabel.RAISED)]
        emitted = _play(first + release + second, release_ms=400, cooldown_ms=500)
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[1].delay_ms, 200)
        assert emitted[1].event is not None
        self.assertEqual(emitted[1].event.stream_timestamp_ms, 1400)

    def test_unavailable_input_does_not_start_a_gesture(self) -> None:
        gesture = TemporalFilter(0, 0, 0)
        update = gesture.update(
            _observation(0, FrameStatus.UNAVAILABLE),
            _prediction(HandRaiseLabel.RAISED),
            source="test",
            clock=_clock,
        )
        self.assertIsNone(update.event)
        self.assertEqual(update.phase.value, "idle")


if __name__ == "__main__":
    unittest.main()
