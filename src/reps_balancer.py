"""REPS: Adaptive probabilistic routing with entropy recycling.

Based on: "REPS: Adaptive Load Balancing Through Recycled Entropy
Packet Spraying" (arXiv:2407.21625).

Key ideas:
- Tasks are routed probabilistically across nodes (packet spraying).
- Probabilities adapt based on observed load feedback.
- "Entropy recycling": the randomness source (entropy) is informed by
  observed load — high-load nodes get lower probability, but never zero
  (maintains exploration/spreading).
- The adaptation loop: observe cost → adjust probabilities → re-normalize
  → next task uses updated distribution.

This implementation models the core idea: a probability distribution over
nodes that adapts to observed costs, maintaining a minimum probability
floor to avoid overloading any single node completely.

The "entropy" aspect: we track the entropy of the probability distribution
H = -sum(p_i * log(p_i)) and recycle it — when entropy drops (distribution
becomes too peaked), the adaptation slows down to maintain diversity.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from src.node import Node
from src.task import Task


class REPSSprayer:
    """Adaptive probabilistic task router with entropy recycling.

    Maintains a probability distribution over nodes. On each task arrival,
    samples a node from the distribution (with a minimum floor to ensure
    spreading). After observing the cost, adjusts probabilities: nodes
    with lower costs get higher probability.

    The adaptation rate is modulated by the distribution's entropy:
    - High entropy (spread out) → faster adaptation
    - Low entropy (peaked) → slower adaptation (recycle entropy)
    """

    def __init__(
        self,
        seed: int = 42,
        num_nodes: int | None = None,
        initial_probs: list[float] | None = None,
        floor: float = 0.05,  # minimum probability per node
        adapt_rate: float = 0.3,  # base learning rate
        sensitivity: float = 1.0,  # how strongly costs affect probs
    ):
        """Parameters
        ----------
        seed: RNG seed for sampling.
        num_nodes: pre-allocate if known.
        initial_probs: optional initial probability distribution (will be
            normalized if not summing to 1).
        floor: minimum probability any node can have (prevents starvation).
        adapt_rate: base learning rate for probability updates.
        sensitivity: multiplier on cost differences when updating probs.
        """
        self._rng = random.Random(seed)
        self._num_nodes = num_nodes
        self._floor = floor
        self._adapt_rate = adapt_rate
        self._sensitivity = sensitivity

        # Probability distribution (will be initialized on first assign)
        self._probs: list[float] = []
        self._node_ids: list[str] = []

        # Running statistics for adaptation
        self._cost_sum: dict[str, float] = {}  # node_id -> sum of costs
        self._cost_count: dict[str, int] = {}  # node_id -> count
        self._avg_costs: dict[str, float] = {}  # node_id -> running avg

        # History for snapshot
        self._samples: list[str] = []
        self._adaptations: int = 0

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Sample a node from the current probability distribution."""
        if not nodes:
            raise ValueError("No nodes available")

        # Initialize distribution on first call
        if not self._probs:
            n = len(nodes)
            self._node_ids = [n.id for n in nodes]
            self._probs = [1.0 / n] * n  # uniform start
            for node in nodes:
                nid = node.id
                self._cost_sum[nid] = 0.0
                self._cost_count[nid] = 0
                self._avg_costs[nid] = 0.0

        # Sample from distribution with floor
        probs = self._enforce_floor(self._probs)
        chosen_idx = self._rng.choices(range(len(nodes)), weights=probs, k=1)[0]
        chosen = nodes[chosen_idx]

        self._samples.append(chosen.id)
        return chosen

    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
        """Update probability distribution based on observed cost."""
        nid = node.id
        if nid not in self._cost_sum:
            return

        # Update running average
        cnt = self._cost_count[nid] + 1
        self._cost_count[nid] = cnt
        self._cost_sum[nid] += actual_cost
        old_avg = self._avg_costs[nid]
        self._avg_costs[nid] = old_avg + (actual_cost - old_avg) / cnt

        # Adapt probabilities
        self._adapt_probabilities()

    def _adapt_probabilities(self) -> None:
        """Adjust probabilities based on observed average costs."""
        n = len(self._probs)

        # Compute mean cost across all nodes (for relative comparison)
        all_avgs = [self._avg_costs[nid] for nid in self._node_ids
                     if self._cost_count[nid] > 0]
        if not all_avgs:
            return  # no data yet

        mean_cost = sum(all_avgs) / len(all_avgs)
        if mean_cost < 1e-9:
            mean_cost = 1e-9

        # Compute new probabilities: inverse cost weighting
        new_probs = []
        for i, nid in enumerate(self._node_ids):
            if self._cost_count[nid] == 0:
                # Node never tried — keep uniform-ish
                new_probs.append(1.0 / n)
            else:
                avg = self._avg_costs[nid]
                # Relative cost (lower = better = higher probability)
                rel_cost = avg / mean_cost
                # Exponential decay: p ~ exp(-sensitivity * (rel_cost - 1))
                # A node with avg_cost = mean_cost gets weight 1.0
                # A node with 2x mean cost gets weight exp(-sensitivity)
                weight = math.exp(-self._sensitivity * (rel_cost - 1.0))
                new_probs.append(max(weight, self._floor))

        # Normalize to sum to 1
        total = sum(new_probs)
        new_probs = [p / total for p in new_probs]

        # Entropy-modulated adaptation rate
        entropy = self._entropy(self._probs)
        max_entropy = math.log(n)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0

        # Recycle: blend old and new based on entropy
        alpha = self._adapt_rate * (0.5 + 0.5 * entropy_ratio)
        alpha = min(alpha, 0.95)  # cap to avoid too-fast adaptation

        self._probs = [(1 - alpha) * old + alpha * new
                       for old, new in zip(self._probs, new_probs)]
        self._adaptations += 1

    def _enforce_floor(self, probs: list[float]) -> list[float]:
        """Ensure all probabilities are at least the floor value."""
        n = len(probs)
        floored = [max(p, self._floor) for p in probs]
        total = sum(floored)
        return [p / total for p in floored]

    @staticmethod
    def _entropy(probs: list[float]) -> float:
        """Compute Shannon entropy of a probability distribution."""
        return -sum(p * math.log(p) for p in probs if p > 0)

    @property
    def snapshot(self) -> dict:
        """Serializable state for gossip or analysis."""
        n = len(self._probs)
        max_entropy = math.log(n) if n > 0 else 1.0
        current_entropy = self._entropy(self._probs)

        return {
            "probs": dict(zip(self._node_ids, self._probs)),
            "avg_costs": dict(self._avg_costs),
            "cost_counts": dict(self._cost_count),
            "entropy": current_entropy,
            "entropy_ratio": current_entropy / max_entropy,
            "adaptations": self._adaptations,
            "samples": self._samples[-100:],
        }

    def merge(self, other: dict) -> None:
        """Merge another snapshot into this sprayer's state."""
        if "avg_costs" not in other or "cost_counts" not in other:
            return

        for nid in other["avg_costs"]:
            if nid not in self._avg_costs:
                continue
            # Merge running averages (weighted by count)
            c1 = self._cost_count.get(nid, 0)
            c2 = other["cost_counts"].get(nid, 0)
            if c1 + c2 > 0:
                self._avg_costs[nid] = (
                    self._avg_costs[nid] * c1 + other["avg_costs"][nid] * c2
                ) / (c1 + c2)
                self._cost_count[nid] = c1 + c2
                self._cost_sum[nid] = self._avg_costs[nid] * self._cost_count[nid]


if __name__ == "__main__":
    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=15.0, memory_cap=15.0, network_cap=8.0),
    ]

    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0)

    from src.truesim import StochasticTrueModel
    model = StochasticTrueModel(seed=42, cost_scale=15.0)

    sprayer = REPSSprayer(seed=42, floor=0.1, adapt_rate=0.5, sensitivity=2.0)

    print("REPS adaptive spraying (3 nodes, 20 tasks):")
    for i in range(20):
        chosen = sprayer.assign(task, nodes)
        cost = model.cost(task, chosen)
        sprayer.on_complete(task, chosen, cost)
        snap = sprayer.snapshot
        print(f"  task-{i}: -> {chosen.id}, cost={cost:.2f}, "
              f"probs={[f'{p:.2f}' for p in snap['probs'].values()]}, "
              f"H={snap['entropy']:.2f} (ratio={snap['entropy_ratio']:.2f})")

    print(f"\nFinal avg costs: {sprayer._avg_costs}")
    print(f"Final snapshot: {sprayer.snapshot}")
