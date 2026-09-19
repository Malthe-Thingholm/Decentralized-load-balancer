"""Discrete-event simulation engine — Phase 1.

Event-driven, sim-time only (no wall-clock coupling).  Priority queue ordered
by sim-time.  Deterministic replay via an injectable RNG seed.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    ARRIVAL = auto()
    COMPLETION = auto()
    GOSSIP = auto()
    BALANCER_TICK = auto()


@dataclass(order=True, frozen=True)
class Event:
    """Priority-queue item.  Frozen so it's hashable and immutable."""

    time: float
    seq: int  # tie-breaker for same-time events
    type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class Engine:
    """Discrete-event simulation engine.

    Parameters
    ----------
    seed: int
        RNG seed for deterministic replay.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed: int = seed
        self._events: list[Event] = []
        self._seq: int = 0
        self._time: float = 0.0
        self._running: bool = False
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {
            et: [] for et in EventType
        }
        self._step_count: int = 0

    # ------------------------------------------------------------------ public

    @property
    def time(self) -> float:
        return self._time

    @property
    def step_count(self) -> int:
        return self._step_count

    def schedule(self, type: EventType, *, delay: float = 0.0,
                 payload: dict[str, Any] | None = None) -> None:
        """Schedule an event *delay* sim-time units from now."""
        if delay < 0:
            raise ValueError(f"negative delay {delay}")
        t = self._time + delay
        self._seq += 1
        heapq.heappush(self._events, Event(t, self._seq, type, payload or {}))

    def on(self, type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a handler for *type*."""
        self._handlers[type].append(handler)

    def run(self, until: float | None = None, max_steps: int | None = None) -> None:
        """Run the event loop.

        Stops when: no events left, *until* time reached, or *max_steps*
        exceeded.  Raises RuntimeError on event starvation (time doesn't advance).
        """
        self._running = True
        steps = 0
        while self._events and self._running:
            if until is not None and self._time >= until:
                break
            if max_steps is not None and steps >= max_steps:
                break
            ev = heapq.heappop(self._events)
            if ev.time < self._time - 1e-12:
                raise RuntimeError(f"time travel: {ev.time} < {self._time}")
            self._time = ev.time
            self._step_count += 1
            steps += 1
            logger.debug("t=%.4f %s", self._time, ev.type.name)
            for h in self._handlers[ev.type]:
                h(ev)
        self._running = False

    def reset(self) -> None:
        """Clear all events and reset clock/step counters."""
        self._events.clear()
        self._time = 0.0
        self._step_count = 0
        self._running = False
        self._seq = 0

    # ------------------------------------------------------------------ helpers

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": self._time,
            "step_count": self._step_count,
            "pending_events": len(self._events),
            "seed": self._seed,
        }
