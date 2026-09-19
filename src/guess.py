"""GuessingModel stub."""

from __future__ import annotations

import random
from typing import Protocol

from src.node import Node
from src.task import Task


class GuessingModel(Protocol):
    """Per-node model for estimating task cost."""

    def estimate(self, task: Task, node: Node) -> float:
        ...

    def snapshot(self) -> dict:
        ...

    def merge(self, other_snapshot: dict) -> None:
        ...


class StochasticGuessingModel:
    """Stochastic per-node guess model with optional gossip state."""

    def __init__(self, seed: int = 0, error_std: float = 0.2) -> None:
        self._rng = random.Random(seed)
        self.error_std = error_std
        self._state: dict = {"version": 1, "observations": []}

    def estimate(self, task: Task, node: Node) -> float:
        # Base estimate: same as TrueModel but with estimation error
        cpu_ratio = task.cpu_req / node.cpu_cap
        mem_ratio = task.memory_req / node.memory_cap
        net_ratio = task.network_req / node.network_cap
        base = (cpu_ratio + mem_ratio + net_ratio) / 3.0
        noise = self._rng.gauss(0, self.error_std)
        return max(base + noise, 0.0)

    def snapshot(self) -> dict:
        return dict(self._state)

    def merge(self, other_snapshot: dict) -> None:
        self._state["version"] += 1
        obs = self._state.get("observations", [])
        other_obs = other_snapshot.get("observations", [])
        self._state["observations"] = obs + other_obs