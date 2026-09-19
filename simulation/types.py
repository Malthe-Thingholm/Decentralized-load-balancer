"""Shared types for the simulation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Capability:
    cpu: float
    memory: float
    network: float


@dataclass(frozen=True)
class Task:
    id: str
    cpu_req: float
    memory_req: float
    network_req: float
    deadline: float | None = None
    priority: int = 0


@dataclass(frozen=True)
class Assignment:
    task: Task
    node_id: str
    cost: float | None = None


@dataclass
class NodeState:
    capability: Capability
    queue: list[Assignment] = field(default_factory=list)
    completed: list[Assignment] = field(default_factory=list)
    total_busy_time: float = 0.0
    current_load: float = 0.0  # sim-time when load clears


# --- Event types ---

EVENT_ARRIVAL = "arrival"
EVENT_COMPLETION = "completion"
EVENT_GOSSIP = "gossip"
EVENT_BALANCER_TICK = "balancer_tick"


@dataclass(order=True, frozen=True)
class Event:
    sim_time: float
    event_type: str = field(compare=False)
    task: Task | None = field(default=None, compare=False)
    node_id: str | None = field(default=None, compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)