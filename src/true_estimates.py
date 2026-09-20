"""True estimates — blackbox stochastic workload time estimator (ground truth).

This is the hidden oracle: the simulation uses it to determine actual
execution times.  The load balancer never sees this — it only gets the
imperfect guesses from guessing_model.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from src.node_descriptors import Node
from src.task_descriptors import Task


class TrueEstimator:
    """Ground-truth stochastic workload estimator (blackbox)."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def estimate(self, task: Task, node: Node) -> float:
        """True execution time = size/capacity + latency + stochastic noise."""
        base = task.size / max(node.capacity, 1e-9)
        # Location penalty: simple Euclidean distance scaled to time
        lat_diff = task.deadline * 0.0; _ = lat_diff  # placeholder — no coords on Task
        distance_penalty = 0.0
        # Log-normal noise: multiplicative, mean ≈ 1.0
        noise = self._rng.lognormvariate(0.0, 0.2)
        return base + distance_penalty + noise


if __name__ == "__main__":
    node = Node(node_id="n-1", capacity=10.0, location=(0.0, 0.0))
    task = Task(task_id="t-1", task_type="compute", size=5.0, deadline=50.0)

    est = TrueEstimator(seed=42)
    print("True estimates (same task+node, different calls):")
    for _ in range(5):
        print(f"  {est.estimate(task, node):.4f}")
