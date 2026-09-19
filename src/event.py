"""Event dataclass and Simulator class with priority queue."""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Generator

EVENT_ARRIVAL = "arrival"
EVENT_COMPLETION = "completion"
EVENT_GOSSIP = "gossip"
EVENT_BALANCER_TICK = "balancer_tick"


@dataclass(order=True, frozen=True)
class Event:
    """Discrete event ordered by sim-time with stable tie-breaking."""

    sim_time: float
    event_type: str = field(compare=False)
    task: Any = field(default=None, compare=False)
    node_id: str | None = field(default=None, compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class Simulator:
    """Event-driven discrete-time simulator with priority queue."""

    def __init__(self, seed: int = 42) -> None:
        self._queue: list[tuple[float, int, Event]] = []
        self._seq = 0
        self.sim_time: float = 0.0
        self._rng = random.Random(seed)
        self._running = False
        self._event_log: list[Event] = []

    def schedule(self, event: Event) -> None:
        """Schedule an event. Ordered by sim_time, then sequence for stability."""
        heapq.heappush(self._queue, (event.sim_time, self._seq, event))
        self._seq += 1

    def next_event(self) -> Event | None:
        """Pop the next event by sim-time. Returns None if queue is empty."""
        if not self._queue:
            return None
        sim_time, _, event = heapq.heappop(self._queue)
        self.sim_time = sim_time
        self._event_log.append(event)
        return event

    def peek_time(self) -> float | None:
        """Return the sim_time of the next scheduled event without popping."""
        if not self._queue:
            return None
        return self._queue[0][0]

    def run(self, until: float | None = None) -> Generator[Event, Any, None]:
        """Process all events until queue empty or time bound reached."""
        self._running = True
        while self._queue:
            if until is not None and self.sim_time >= until:
                break
            event = self.next_event()
            if event is None:
                break
            yield event
        self._running = False

    def reset(self) -> None:
        """Reset simulator state for deterministic replay."""
        self._queue.clear()
        self._seq = 0
        self.sim_time = 0.0
        self._event_log.clear()

    @property
    def event_log(self) -> list[Event]:
        return list(self._event_log)

    @property
    def queue_depth(self) -> int:
        return len(self._queue)