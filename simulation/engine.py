"""Event-driven discrete simulation engine — Phase 1 core."""

from __future__ import annotations

import heapq
import random
from typing import Callable

from simulation.types import (
    EVENT_ARRIVAL,
    EVENT_BALANCER_TICK,
    EVENT_COMPLETION,
    EVENT_GOSSIP,
    Assignment,
    Capability,
    Event,
    NodeState,
    Task,
)

Handler = Callable[[Event, "Engine"], None]


class Engine:
    """Discrete-event simulation engine.

    Events are stored in a min-heap keyed by ``sim_time``.  Deterministic
    replay is achieved by seeding the internal RNG — the same seed + same
    event schedule always produces the same execution trace.
    """

    def __init__(self, seed: int = 42) -> None:
        self._queue: list[tuple[float, int, Event]] = []
        self._seq = 0  # tie-breaker for equal sim_times
        self.time = 0.0
        self.seed = seed
        self._rng = random.Random(seed)
        self._handlers: dict[str, list[Handler]] = {
            EVENT_ARRIVAL: [],
            EVENT_COMPLETION: [],
            EVENT_GOSSIP: [],
            EVENT_BALANCER_TICK: [],
        }
        self.nodes: dict[str, NodeState] = {}
        self.log: list[Event] = []  # replay trace

    # --- event scheduling ---

    def schedule(self, event: Event) -> None:
        """Push an event onto the queue.  Order preserved for equal sim_times."""
        heapq.heappush(self._queue, (event.sim_time, self._seq, event))
        self._seq += 1

    def on(self, event_type: str, handler: Handler) -> None:
        """Register a callback for *event_type*."""
        self._handlers[event_type].append(handler)

    # --- node management ---

    def add_node(self, node_id: str, capability: Capability) -> None:
        self.nodes[node_id] = NodeState(capability=capability)

    # --- execution ---

    def run(self, until: float = float("inf")) -> None:
        """Process events until the queue is empty or *until* sim-time."""
        while self._queue:
            sim_time, _, event = self._queue[0]
            if sim_time > until:
                break
            heapq.heappop(self._queue)
            self.time = sim_time
            self.log.append(event)
            for handler in self._handlers[event.event_type]:
                handler(event, self)

    def step(self) -> Event | None:
        """Process a single event and return it.  Returns None when empty."""
        if not self._queue:
            return None
        sim_time, _, event = heapq.heappop(self._queue)
        self.time = sim_time
        self.log.append(event)
        for handler in self._handlers[event.event_type]:
            handler(event, self)
        return event

    # --- deterministic replay ---

    def snapshot(self) -> dict:
        """Serialisable state for deterministic replay."""
        return {
            "seed": self.seed,
            "time": self.time,
            "queue_len": len(self._queue),
            "nodes": {
                nid: {
                    "queue_len": len(s.queue),
                    "completed": len(s.completed),
                    "total_busy_time": s.total_busy_time,
                }
                for nid, s in self.nodes.items()
            },
        }

    def rng_int(self, lo: int, hi: int) -> int:
        """Pull from the deterministic RNG (used by gossip etc.)."""
        return self._rng.randint(lo, hi)