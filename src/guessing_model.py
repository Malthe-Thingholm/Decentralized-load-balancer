"""Guessing model — in-simulation imperfect estimate used by load balancer.

The load balancer sees only this estimator's output.  It uses the same
base formula as true_estimates.py but adds systematic bias and
higher-variance noise to model imperfect observability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from src.node_descriptors import Node
from src.task_descriptors import Task


class GuessingModel:
    """Imperfect workload-time predictor (what the balancer actually uses)."""

    def __init__(self, bias: float = 0.15, noise_std: float = 0.35, seed: int | None = None):
        self._bias = bias            # systematic overestimate tendency
        self._noise_std = noise_std  # higher variance than ground truth
        self._rng = random.Random(seed)

    def estimate(self, task: Task, node: Node) -> float:
        """Guess execution time = biased base + heavy noise."""
        base = task.size / max(node.capacity, 1e-9)
        # Systematic bias: balancer overestimates by bias fraction
        biased = base * (1.0 + self._bias)
        # Gaussian noise (heavier than true estimator's log-normal)
        noise = self._rng.gauss(0.0, self._noise_std * base)
        return max(0.0, biased + noise)


if __name__ == "__main__":
    node = Node(node_id="n-1", capacity=10.0, location=(0.0, 0.0))
    task = Task(task_id="t-1", task_type="compute", size=5.0, deadline=50.0)

    model = GuessingModel(bias=0.15, noise_std=0.35, seed=42)
    print("Guessed estimates (same task+node, different calls):")
    for _ in range(5):
        print(f"  {model.estimate(task, node):.4f}")
