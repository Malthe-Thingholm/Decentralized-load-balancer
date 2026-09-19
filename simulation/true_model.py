"""Stochastic black-box TrueModel — Phase 1 stub."""

from __future__ import annotations

import random

from simulation.types import Capability, Task


class TrueModel:
    """Black-box stochastic cost model.

    ``cost(task, node)`` returns a float drawn from a distribution whose
    parameters depend on the task/node capability mismatch.  Called exactly
    once per assignment — the :class:`AssertedTrueModel` wrapper enforces
    this invariant.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def cost(self, task: Task, node_cap: Capability) -> float:
        """Stochastic execution cost for *task* on a node with *node_cap*."""
        cpu_match = task.cpu_req / max(node_cap.cpu, 1e-9)
        mem_match = task.memory_req / max(node_cap.memory, 1e-9)
        net_match = task.network_req / max(node_cap.network, 1e-9)
        # Base cost driven by the worst resource mismatch, plus noise.
        base = max(cpu_match, mem_match, net_match) * 10.0
        noise = self._rng.uniform(0.8, 1.2)
        return max(base * noise, 0.1)


class AssertedTrueModel:
    """Wrapper that asserts ``cost`` is called exactly once per assignment."""

    def __init__(self, inner: TrueModel) -> None:
        self._inner = inner
        self._called_for: dict[str, bool] = {}

    def cost(self, task: Task, node_cap: Capability) -> float:
        if task.id in self._called_for:
            raise AssertionError(
                f"TrueModel.cost called multiple times for task {task.id!r}"
            )
        self._called_for[task.id] = True
        return self._inner.cost(task, node_cap)

    def assert_consumed(self, task_ids: list[str]) -> None:
        """Raise if any *task_ids* never had cost() called."""
        missing = [tid for tid in task_ids if tid not in self._called_for]
        if missing:
            raise AssertionError(
                f"TrueModel.cost never called for tasks: {missing}"
            )

    def reset(self) -> None:
        self._called_for.clear()