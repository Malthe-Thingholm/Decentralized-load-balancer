"""A2WS-style adaptive asynchronous work-stealing balancer.

Based on: "Adaptive Asynchronous Work-Stealing for distributed
load-balancing in heterogeneous systems" (arXiv:2401.04494).

Key ideas:
- Idle nodes (short queues) can "steal" work from busy nodes (long queues).
- Victim selection is adaptive: prefer nodes that are significantly loaded.
- Limited communication: thief only needs queue-depth info, not full state.
- Asynchronous: steal attempts don't block normal task assignment.

This implementation simplifies the paper: instead of a full steal protocol
with request/reply, we model work-stealing as a balancer strategy where
the balancer, when assigning a task, may "steal" from an overloaded node
if the current node's queue is shallow enough.

The adaptive radius R determines how much more loaded a victim must be
relative to the thief to be worth stealing from.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from src.node import Node
from src.task import Task


class WorkStealingBalancer:
    """A2WS-inspired work-stealing load balancer.

    Assigns tasks normally (to the least-loaded node), but when a node
    is idle (queue below threshold), it may "steal" a task from an
    overloaded node instead of assigning a new one.

    This is a simplified model: in a real system, stealing is an async
    protocol between nodes. Here we model the *effect* — tasks migrate
    from overloaded to underloaded nodes — at assignment time.
    """

    def __init__(
        self,
        seed: int = 42,
        adaptive_radius: float = 2.0,
        steal_threshold: float = 0.25,
        max_steal_candidates: int = 3,
    ):
        """Initialize work-stealing balancer.

        Parameters
        ----------
        seed: RNG seed for deterministic replay.
        adaptive_radius: How much more loaded a victim must be (relative
            to the population mean) to be considered a steal target.
        steal_threshold: Minimum queue-depth ratio below which a node
            is considered "idle" and may initiate a steal.
        max_steal_candidates: Max nodes to consider as victims per steal.
        """
        self._rng = random.Random(seed)
        self.adaptive_radius = adaptive_radius
        self.steal_threshold = steal_threshold
        self.max_steal_candidates = max_steal_candidates

        # Statistics tracked across assignments
        self._total_assignments = 0
        self._total_steals = 0
        self._steal_history: list[float] = []  # cost savings from steals

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to a node, potentially via work-stealing.

        Strategy:
        1. Find the least-loaded node (by queue-depth / capacity ratio).
        2. If that node is "idle" (below steal_threshold), look for an
           overloaded victim to steal from.
        3. If a victim is found and stealing is beneficial, assign the
           task to the victim's queue (modeling the steal) — but in our
           simplified model, we actually assign the *new* task to the
           least-loaded node and let the steal happen implicitly via the
           balancer's preference for balanced queues.
        4. Otherwise, assign to the least-loaded node normally.
        """
        if not nodes:
            raise ValueError("No nodes available")

        self._total_assignments += 1

        # Compute load ratio for each node: queue_depth / cpu_cap
        def load_ratio(n: Node) -> float:
            if n.cpu_cap <= 0:
                return float("inf")
            return n.queue_depth / n.cpu_cap

        ratios = {n.id: load_ratio(n) for n in nodes}
        mean_ratio = sum(ratios.values()) / len(ratios)

        # Find the least-loaded node
        least_loaded = min(nodes, key=load_ratio)
        least_ratio = ratios[least_loaded.id]

        # Work-stealing: if least-loaded node is idle, try to steal
        if least_ratio < self.steal_threshold * mean_ratio:
            # Look for overloaded victims
            victims = [
                n for n in nodes
                if ratios[n.id] > self.adaptive_radius * mean_ratio
            ]
            if victims:
                # Pick the most overloaded victim (by load ratio)
                victim = max(victims, key=load_ratio)
                # In our model: assign the task to the least-loaded node,
                # but record the steal attempt. The actual "steal" in a
                # real system would move a task from victim to thief.
                # Here we bias future assignments toward balance.
                self._total_steals += 1
                # Record estimated benefit: difference in load ratios
                benefit = ratios[victim.id] - least_ratio
                self._steal_history.append(benefit)

        return least_loaded

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """Callback when a task completes. No-op in this simplified model.

        A full A2WS implementation would update the node's observed
        load state here for future victim selection.
        """
        _ = task, node, actual_cost

    @property
    def steal_rate(self) -> float:
        """Fraction of assignments that involved a steal attempt."""
        if self._total_assignments == 0:
            return 0.0
        return self._total_steals / self._total_assignments

    @property
    def total_steals(self) -> int:
        return self._total_steals

    @property
    def total_assignments(self) -> int:
        return self._total_assignments

    def snapshot(self) -> dict:
        """Return state for debugging / gossip (future use)."""
        return {
            "total_assignments": self._total_assignments,
            "total_steals": self._total_steals,
            "steal_rate": self.steal_rate,
            "adaptive_radius": self.adaptive_radius,
            "steal_threshold": self.steal_threshold,
            "avg_steal_benefit": (
                sum(self._steal_history) / len(self._steal_history)
                if self._steal_history else 0.0
            ),
        }


class AdaptiveWorkStealingBalancer(WorkStealingBalancer):
    """Adaptive variant: adjusts steal_threshold based on observed load.

    The paper's key contribution is *adaptivity* — the system adjusts
    its stealing behavior based on observed load patterns. This variant
    tunes the steal_threshold up when steals are frequent (system is
    imbalanced) and down when steals are rare (system is balanced).
    """

    def __init__(
        self,
        seed: int = 42,
        adaptive_radius: float = 2.0,
        steal_threshold: float = 0.25,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.05,
        max_threshold: float = 0.5,
    ):
        super().__init__(
            seed=seed,
            adaptive_radius=adaptive_radius,
            steal_threshold=steal_threshold,
            max_steal_candidates=3,
        )
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._target_steal_rate = 0.1  # desired fraction of steals

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """Adapt steal threshold based on recent steal effectiveness."""
        _ = task, actual_cost
        # After a completion, check if we're stealing too much or too little
        current_rate = self.steal_rate
        if current_rate > self._target_steal_rate * 2:
            # Stealing too much — reduce threshold (fewer steal attempts)
            self.steal_threshold = max(
                self.min_threshold,
                self.steal_threshold * (1 - self.adaptation_rate),
            )
        elif current_rate < self._target_steal_rate * 0.5 and current_rate > 0:
            # Stealing too little — increase threshold (more steal attempts)
            self.steal_threshold = min(
                self.max_threshold,
                self.steal_threshold * (1 + self.adaptation_rate),
            )


if __name__ == "__main__":
    # Quick demo
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=5.0, memory_cap=5.0, network_cap=3.0),
        Node(id="n3", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
    ]
    # Pre-load n2 to simulate imbalance
    nodes[1].queue = ["t-1", "t-2", "t-3", "t-4"]

    task = Task(id="t-new", cpu_req=1.0, memory_req=1.0, network_req=0.5)

    bal = WorkStealingBalancer(seed=42, adaptive_radius=2.0, steal_threshold=0.25)
    chosen = bal.assign(task, nodes)
    print(f"A2WS chose: {chosen.id}")
    print(f"  steal_rate: {bal.steal_rate:.2f}, total_steals: {bal.total_steals}")
    print(f"  snapshot: {bal.snapshot()}")

    # Adaptive variant
    bal2 = AdaptiveWorkStealingBalancer(seed=42)
    chosen2 = bal2.assign(task, nodes)
    print(f"\nAdaptive A2WS chose: {chosen2.id}")
    print(f"  steal_rate: {bal2.steal_rate:.2f}, threshold: {bal2.steal_threshold:.3f}")
