"""TrueModel interface and StochasticTrueModel stub."""

from __future__ import annotations

import random
from typing import Protocol

from src.node import Node
from src.task import Task


class TrueModel(Protocol):
    """Stochastic black-box cost model. Called once per assignment."""

    def cost(self, task: Task, node: Node) -> float:
        ...


def assert_cost(func):
    """Assertion wrapper for TrueModel.cost — validates non-negative output."""

    def wrapper(*args, **kwargs) -> float:
        result = func(*args, **kwargs)
        assert result >= 0, f"TrueModel cost must be >= 0, got {result}"
        return result

    return wrapper


class StochasticTrueModel:
    """Stochastic TrueModel: cost = base_load + Gaussian noise."""

    def __init__(self, seed: int = 42, noise_std: float = 0.1) -> None:
        self._rng = random.Random(seed)
        self.noise_std = noise_std

    @assert_cost
    def cost(self, task: Task, node: Node) -> float:
        # Base cost: ratio of demand to capacity, weighted equally
        cpu_ratio = task.cpu_req / node.cpu_cap
        mem_ratio = task.memory_req / node.memory_cap
        net_ratio = task.network_req / node.network_cap
        base_cost = (cpu_ratio + mem_ratio + net_ratio) / 3.0

        # Stochastic noise
        noise = self._rng.gauss(0, self.noise_std)
        cost = base_cost + noise

        # Clamp to non-negative (assert_cost also enforces this)
        return max(cost, 0.0)