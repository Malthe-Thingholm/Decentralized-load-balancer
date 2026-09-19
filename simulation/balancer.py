"""Balancer interface + baseline strategies — Phase 1."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from simulation.types import Assignment, Capability, Task


class Balancer(ABC):
    """Configurable assignment strategy."""

    @abstractmethod
    def assign(self, task: Task, nodes: dict[str, Capability]) -> str:
        """Return the node_id to assign *task* to."""

    @abstractmethod
    def on_complete(self, task: Task, node_id: str, actual_cost: float) -> None:
        """Callback when a task finishes on *node_id*."""


class RandomBalancer(Balancer):
    """Assigns tasks uniformly at random."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def assign(self, task: Task, nodes: dict[str, Capability]) -> str:
        return self._rng.choice(list(nodes.keys()))

    def on_complete(self, task: Task, node_id: str, actual_cost: float) -> None:
        pass


class RoundRobinBalancer(Balancer):
    """Cycles through nodes in order."""

    def __init__(self) -> None:
        self._idx = 0

    def assign(self, task: Task, nodes: dict[str, Capability]) -> str:
        ids = list(nodes.keys())
        node_id = ids[self._idx % len(ids)]
        self._idx += 1
        return node_id

    def on_complete(self, task: Task, node_id: str, actual_cost: float) -> None:
        pass


class ShortestQueueBalancer(Balancer):
    """Assigns to the node with the fewest queued tasks."""

    def __init__(self, node_queues: dict[str, list]) -> None:
        self._queues = node_queues

    def assign(self, task: Task, nodes: dict[str, Capability]) -> str:
        return min(self._queues, key=lambda nid: len(self._queues[nid]))

    def on_complete(self, task: Task, node_id: str, actual_cost: float) -> None:
        pass