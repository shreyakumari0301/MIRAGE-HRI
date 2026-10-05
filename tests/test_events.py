"""Event schema and simulated-robot tests. No camera is required."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from events.consumer import RobotConsumer  # noqa: E402
from events.log import JsonlLog  # noqa: E402
from events.schema import (  # noqa: E402
    EventStatus,
    build_interaction_event,
    raw_frame_record,
)
from perception.hand_raise import predict_hand_raise  # noqa: E402
from perception.status import FrameStatus, Landmark, PoseObservation  # noqa: E402

MOMENT = datetime(2026, 10, 5, 12, 30, 1, 250000, tzinfo=timezone.utc)
SOURCE = "webcam/hand_raise_rule"


def _clock() -> datetime:
    return MOMENT


def _point(y: float, visibility: float | None) -> Landmark:
    return Landmark(x=0.5, y=y, z=0.0, visibility=visibility)


def _pose() -> list[Landmark | None]:
    return [Landmark(0.5, 0.55, 0.0, 0.2) for _ in range(33)]


def _prediction(points: list[Landmark | None]):
    return predict_hand_raise(tuple(points), min_visibility=0.5, raise_margin=0.08)


def _observation(
    status: FrameStatus,
    points: list[Landmark | None] | None,
    timestamp_ms: int = 842,
) -> PoseObservation:
    landmarks = None if points is None else tuple(points)
    return PoseObservation(
        status=status,
        landmarks=landmarks,
        visible_count=0 if landmarks is None else 4,
        message="test",
        timestamp_ms=timestamp_ms,
    )


def _raised_points() -> list[Landmark | None]:
    points = _pose()
    points[12] = _point(0.50, 0.91)
    points[16] = _point(0.20, 0.87)
    return points


class EventSchemaTests(unittest.TestCase):
    def test_raised_event_has_timestamp_score_and_source(self) -> None:
        points = _raised_points()
        prediction = _prediction(points)
        event = build_interaction_event(
            _observation(FrameStatus.DETECTED, points),
            prediction,
            source=SOURCE,
            clock=_clock,
        )
        assert event is not None
        self.assertEqual(event.status, EventStatus.DETECTED)
        self.assertEqual(event.cue, "hand_raised")
        self.assertEqual(event.raw_label, "raised")
        self.assertEqual(event.timestamp, "2026-10-05T12:30:01.250Z")
        self.assertEqual(event.stream_timestamp_ms, 842)
        self.assertEqual(event.source, SOURCE)
        self.assertEqual(event.detector_score_kind, "uncalibrated_clearance")
        self.assertEqual(event.detector_score, 0.3)
        payload = json.loads(event.to_json())
        self.assertEqual(
            set(payload),
            {
                "record",
                "schema_version",
                "timestamp",
                "stream_timestamp_ms",
                "cue",
                "status",
                "raw_label",
                "detector_score",
                "detector_score_kind",
                "source",
            },
        )
        self.assertNotIn("probability", event.to_json())
        self.assertNotIn("confidence", event.to_json())

    def test_insufficient_landmarks_are_unknown_without_a_score(self) -> None:
        prediction = _prediction(_pose())
        event = build_interaction_event(
            _observation(FrameStatus.UNKNOWN, _pose()),
            prediction,
            source=SOURCE,
            clock=_clock,
        )
        assert event is not None
        self.assertEqual(event.status, EventStatus.UNKNOWN)
        self.assertIsNone(event.detector_score)
        self.assertEqual(event.raw_label, "insufficient")

    def test_unavailable_pose_does_not_invent_a_raise(self) -> None:
        points = _raised_points()
        event = build_interaction_event(
            _observation(FrameStatus.UNAVAILABLE, None, timestamp_ms=9),
            _prediction(points),
            source=SOURCE,
            clock=_clock,
        )
        assert event is not None
        self.assertEqual(event.status, EventStatus.UNAVAILABLE)
        self.assertIsNone(event.detector_score)
        self.assertEqual(event.stream_timestamp_ms, 9)

    def test_not_raised_stays_out_of_the_event_stream(self) -> None:
        points = _pose()
        points[11] = _point(0.40, 0.95)
        points[15] = _point(0.70, 0.93)
        prediction = _prediction(points)
        observation = _observation(FrameStatus.DETECTED, points, timestamp_ms=15)
        self.assertIsNone(
            build_interaction_event(observation, prediction, source=SOURCE, clock=_clock)
        )
        raw = raw_frame_record(observation, prediction, clock=_clock)
        self.assertEqual(raw.raw_label, "not_raised")
        self.assertEqual(raw.stream_timestamp_ms, 15)
        self.assertLess(raw.detector_score, 0)


class ConsumerTests(unittest.TestCase):
    def _event(self, status_frame: FrameStatus = FrameStatus.DETECTED):
        points = _raised_points()
        return build_interaction_event(
            _observation(status_frame, points),
            _prediction(points),
            source=SOURCE,
            clock=_clock,
        )

    def test_detected_event_reaches_the_consumer(self) -> None:
        event = self._event()
        assert event is not None
        lines: list[str] = []
        consumer = RobotConsumer(writer=lines.append)
        self.assertTrue(consumer.receive(event))
        self.assertEqual(consumer.events, [event])
        self.assertIn("2026-10-05T12:30:01.250Z", lines[0])
        self.assertIn("uncalibrated_clearance", lines[0])
        self.assertIn("hand_raised", lines[0])
        self.assertIn("uncalibrated", consumer.overlay_line or "")

    def test_unknown_and_unavailable_get_no_robot_response(self) -> None:
        unknown = build_interaction_event(
            _observation(FrameStatus.UNKNOWN, _pose()),
            _prediction(_pose()),
            source=SOURCE,
            clock=_clock,
        )
        unavailable = build_interaction_event(
            _observation(FrameStatus.UNAVAILABLE, None),
            _prediction(_pose()),
            source=SOURCE,
            clock=_clock,
        )
        assert unknown is not None and unavailable is not None
        lines: list[str] = []
        consumer = RobotConsumer(writer=lines.append)
        self.assertFalse(consumer.receive(unknown))
        self.assertFalse(consumer.receive(unavailable))
        self.assertEqual(lines, [])
        self.assertEqual(consumer.events, [])

    def test_log_keeps_raw_and_event_records(self) -> None:
        points = _raised_points()
        prediction = _prediction(points)
        observation = _observation(FrameStatus.DETECTED, points)
        event = build_interaction_event(observation, prediction, source=SOURCE, clock=_clock)
        assert event is not None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            log = JsonlLog(path)
            try:
                log.write(raw_frame_record(observation, prediction, clock=_clock).to_dict())
                log.write(event.to_dict())
            finally:
                log.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["record"] for row in rows], ["raw", "event"])
        self.assertEqual(rows[0]["raw_label"], "raised")
        self.assertEqual(rows[1]["status"], "detected")
        self.assertEqual(rows[1]["timestamp"], "2026-10-05T12:30:01.250Z")


if __name__ == "__main__":
    unittest.main()
