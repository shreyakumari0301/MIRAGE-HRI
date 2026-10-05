"""Turn flickering raw hand-raise labels into one gesture event.

Raw labels stay untouched. This filter only decides when a robot event starts
and ends.

Timing uses the pose stream clock, in milliseconds:

- Arming starts on ``raised`` and cancels on any other label.
- A start event is emitted once ``activation_ms`` of continuous ``raised``
  has elapsed. ``delay_ms`` is that wait, measured from the first frame of
  the run that succeeded.
- The gesture stays active until a non-raised label lasts ``release_ms``.
- After the end, new raises are ignored until ``cooldown_ms`` has passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from events.schema import Clock, InteractionEvent, build_interaction_event
from perception.hand_raise import HandRaiseLabel, HandRaisePrediction
from perception.status import FrameStatus, PoseObservation


class FilterPhase(str, Enum):
    IDLE = "idle"
    ARMING = "arming"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class FilterUpdate:
    """One filter step. ``event`` is set only on a gesture start."""

    phase: FilterPhase
    event: InteractionEvent | None
    edge: str | None
    delay_ms: int | None
    overlay: str

    def to_log(self) -> dict[str, object] | None:
        if self.edge is None:
            return None
        payload: dict[str, object] = {
            "record": "filtered",
            "edge": self.edge,
            "delay_ms": self.delay_ms,
            "phase": self.phase.value,
        }
        if self.event is not None:
            payload.update(self.event.to_dict())
            payload["record"] = "filtered"
            payload["edge"] = self.edge
            payload["delay_ms"] = self.delay_ms
        return payload


class TemporalFilter:
    """Stateful gesture gate. One instance follows one webcam stream."""

    def __init__(self, activation_ms: int, release_ms: int, cooldown_ms: int) -> None:
        if activation_ms < 0 or release_ms < 0 or cooldown_ms < 0:
            raise ValueError("filter durations must be zero or positive")
        self.activation_ms = activation_ms
        self.release_ms = release_ms
        self.cooldown_ms = cooldown_ms
        self.phase = FilterPhase.IDLE
        self._arm_started_ms: int | None = None
        self._active_started_ms: int | None = None
        self._release_started_ms: int | None = None
        self._cooldown_until_ms: int | None = None
        self._last_ms: int | None = None

    def update(
        self,
        observation: PoseObservation,
        prediction: HandRaisePrediction,
        *,
        source: str,
        clock: Clock,
    ) -> FilterUpdate:
        now = observation.timestamp_ms
        if self._last_ms is not None and now < self._last_ms:
            return self._view()
        self._last_ms = now
        label = prediction.label
        if observation.status is FrameStatus.UNAVAILABLE:
            label = HandRaiseLabel.INSUFFICIENT

        if self.phase is FilterPhase.COOLDOWN and self._cooldown_until_ms is not None:
            if now < self._cooldown_until_ms:
                return self._view()
            self.phase = FilterPhase.IDLE
            self._cooldown_until_ms = None

        if self.phase in {FilterPhase.IDLE, FilterPhase.ARMING}:
            return self._arm(now, label, observation, prediction, source, clock)
        return self._hold(now, label)

    def _arm(
        self,
        now: int,
        label: HandRaiseLabel,
        observation: PoseObservation,
        prediction: HandRaisePrediction,
        source: str,
        clock: Clock,
    ) -> FilterUpdate:
        if label is not HandRaiseLabel.RAISED:
            self.phase = FilterPhase.IDLE
            self._arm_started_ms = None
            return self._view()
        if self._arm_started_ms is None:
            self._arm_started_ms = now
            self.phase = FilterPhase.ARMING
        delay = now - self._arm_started_ms
        if delay < self.activation_ms:
            return self._view()
        event = build_interaction_event(
            observation,
            prediction,
            source=source,
            clock=clock,
        )
        self.phase = FilterPhase.ACTIVE
        self._active_started_ms = self._arm_started_ms
        self._arm_started_ms = None
        self._release_started_ms = None
        return FilterUpdate(
            phase=self.phase,
            event=event,
            edge="start",
            delay_ms=delay,
            overlay=self._overlay(),
        )

    def _hold(self, now: int, label: HandRaiseLabel) -> FilterUpdate:
        if label is HandRaiseLabel.RAISED:
            self._release_started_ms = None
            return self._view()
        if self._release_started_ms is None:
            self._release_started_ms = now
        held = now - self._release_started_ms
        if held < self.release_ms:
            return self._view()
        started = self._active_started_ms
        delay = None if started is None else now - started
        self.phase = FilterPhase.COOLDOWN
        self._cooldown_until_ms = now + self.cooldown_ms
        self._active_started_ms = None
        self._release_started_ms = None
        return FilterUpdate(
            phase=self.phase,
            event=None,
            edge="end",
            delay_ms=delay,
            overlay=self._overlay(),
        )

    def _view(self) -> FilterUpdate:
        return FilterUpdate(
            phase=self.phase,
            event=None,
            edge=None,
            delay_ms=None,
            overlay=self._overlay(),
        )

    def _overlay(self) -> str:
        return f"filter: {self.phase.value}"
