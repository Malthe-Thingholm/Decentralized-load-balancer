"""TrueModel stub — stochastic black-box cost function.

Phase 1: returns a sampled cost for a (task, node) pair.  Real implementation
will be swapped in later; this stub is enough for engine wiring and tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    requirements: dict[str, float]
    task_id: str = ""
    deadline: float | None = None
    priority: int = 0


@dataclass(frozen=True)
class Node:
    capability: dict[str, float]
    node_id: str = ""
    queue: list[str] = field(default_factory=list, repr=False)


class TrueModel:
    """Stochastic cost model.

    Cost = base_latency + noise, where base_latency is derived from
    task requirements vs node capability.  Noise is Gaussian, seedable.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def cost(self, task: Task, node: Node) -> float:
        """Return stochastic cost for assigning *task* to *node*."""
        base = self._base_latency(task, node)
        noise = self._rng.gauss(0, 0.1 * base) if base > 0 else 0.0
        return max(base + noise, 1e-6)

    def _base_latency(self, task: Task, node: Node) -> float:
        # Simple heuristic: ratio of requirement to capability, max over dims
        ratios = []
        for k, v in task.requirements.items():
            cap = node.capability.get(k, 0.0)
            ratios.append(v / cap if cap > 0 else float("inf"))
        return max(ratios) if ratios else 1.0
