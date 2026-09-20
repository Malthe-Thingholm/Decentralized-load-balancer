"""Mean-Field Multi-Armed Bandit load balancer.

Based on: "Decentralized Task Offloading and Load-Balancing for
Mobile Edge Computing in Dense Networks" (arXiv:2407.00080).

Key ideas:
- Each task arrival is a bandit decision: which node to assign to?
- Arms = nodes. Reward = negative observed cost (lower cost = higher reward).
- UCB1: upper confidence bound balances exploitation (avg reward) and
  exploration (uncertainty bonus that shrinks as arm is pulled more).
- Mean-field term captures aggregate population behavior — we approximate
  this as the global average cost across all nodes, which each bandit
  uses as a baseline for comparison.

This implementation simplifies the paper:
- No explicit mean-field ODE; we use the empirical average as proxy.
- Single bandit per balancer (not per-device); the balancer decides
  for all incoming tasks.
- UCB1 with rewards in [0, 1] range (normalized costs).
"""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

from src.guess import StochasticGuessingModel
from src.node import Node
from src.task import Task


class MeanFieldMABBalancer:
    """UCB1 bandit balancer for task→node assignment.

    For each incoming task, computes UCB1 for each node:
        UCB(a) = Q(a) + sqrt(2 * ln(t) / N(a))
    where:
        Q(a) = average reward for arm a (negative cost, normalized)
        N(a) = number of times arm a was pulled
        t = total pulls across all arms
    Picks the arm with highest UCB.

    Attributes:
        snapshot_dict: serializable state for gossip/decentralization.
    """

    def __init__(
        self,
        seed: int = 42,
        num_nodes: int | None = None,  # set on first assign if known
        normalize_costs: bool = True,
        min_cost: float = 0.0,
        max_cost: float = 10.0,
    ):
        """Parameters
        ----------
        seed: RNG seed (not used in UCB1 itself; deterministic given history).
        num_nodes: optional hint for pre-allocating per-node stats.
        normalize_costs: if True, clip costs to [min_cost, max_cost] and
            map to reward = 1 - (cost - min) / (max - min).
        min_cost, max_cost: cost range for normalization. Updated online
            if normalize_costs is True and actual costs fall outside.
        """
        self._seed = seed
        self._num_nodes = num_nodes
        self._normalize = normalize_costs
        self._min_cost = min_cost
        self._max_cost = max_cost

        # Per-arm statistics: keyed by node id
        self._arm_stats: dict[str, dict] = {}

        # Global counters
        self._total_pulls = 0

        # For mean-field approximation: running average of all costs
        self._cost_sum = 0.0
        self._cost_count = 0
        self._mean_cost = 0.0

        # History for snapshot
        self._assignments: list[str] = []
        self._steals: int = 0

        self._guessing_model: StochasticGuessingModel | None = None

    def set_guess_model(self, model: StochasticGuessingModel) -> None:
        """Store a reference to the guessing model for cost estimation."""
        self._guessing_model = model

    def _get_reward(self, task: Task, node: Node, actual_cost: float) -> float:
        """Compute reward for assigning *task* to *node*.

        If a guessing model is available, uses the guessed cost (what the
        balancer would have estimated) rather than the true cost — this
        makes the balancer learn from its own (potentially inaccurate)
        estimates, matching the black-box guessing setup.
        """
        if self._guessing_model is not None:
            estimated_cost = self._guessing_model.cost(task, node)
        else:
            estimated_cost = actual_cost

        if self._normalize:
            self._min_cost = min(self._min_cost, estimated_cost)
            self._max_cost = max(self._max_cost, estimated_cost)
            rng = self._max_cost - self._min_cost
            if rng > 0:
                normalized_cost = (estimated_cost - self._min_cost) / rng
            else:
                normalized_cost = 0.0
        else:
            normalized_cost = estimated_cost

        return 1.0 - normalized_cost

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to the node with highest UCB1 score."""
        if not nodes:
            raise ValueError("No nodes available")

        t = self._total_pulls

        # Ensure all nodes have stats entries (cold start: first pull gets
        # infinite UCB, so it's always selected first).
        for node in nodes:
            nid = node.id
            if nid not in self._arm_stats:
                self._arm_stats[nid] = {"sum_reward": 0.0, "count": 0}

        # Cold start: if any arm has 0 pulls, pick it (UCB = infinity)
        for node in nodes:
            if self._arm_stats[node.id]["count"] == 0:
                chosen = node
                break
        else:
            # All arms have been pulled at least once — compute UCB
            best_node = None
            best_ucb = -float("inf")

            for node in nodes:
                stats = self._arm_stats[node.id]
                n_a = stats["count"]
                q_a = stats["sum_reward"] / n_a if n_a > 0 else 0.0

                if t > 0 and n_a > 0:
                    exploration = math.sqrt(2.0 * math.log(t) / n_a)
                else:
                    exploration = float("inf")

                ucb = q_a + exploration
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_node = node

            chosen = best_node

        assert chosen is not None
        self._assignments.append(chosen.id)
        return chosen

    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
        """Update bandit statistics with observed cost.

        Uses the guessing model (if set) to compute the reward from the
        estimated cost rather than the true cost — so the balancer learns
        from its own estimates, matching the black-box setup.
        """
        if node.id not in self._arm_stats:
            return

        reward = self._get_reward(task, node, actual_cost)

        stats = self._arm_stats[node.id]
        stats["sum_reward"] += reward
        stats["count"] += 1
        self._total_pulls += 1

        # Update mean-field estimate using estimated cost
        if self._guessing_model is not None:
            estimated_cost = self._guessing_model.cost(task, node)
        else:
            estimated_cost = actual_cost

        self._cost_sum += estimated_cost
        self._cost_count += 1
        self._mean_cost = self._cost_sum / self._cost_count

    @property
    def snapshot(self) -> dict:
        """Serializable state for gossip or decentralized exchange."""
        return {
            "total_pulls": self._total_pulls,
            "mean_cost": self._mean_cost,
            "cost_count": self._cost_count,
            "arm_stats": {
                nid: {"sum_reward": s["sum_reward"], "count": s["count"]}
                for nid, s in self._arm_stats.items()
            },
            "assignments": self._assignments[-100:],  # last 100
            "steals": self._steals,
        }

    def merge(self, other: dict) -> None:
        """Merge another snapshot into this balancer's state.

        Used for decentralized gossip: when two balancers exchange state,
        they merge their bandit statistics.
        """
        if "total_pulls" in other:
            self._total_pulls += other["total_pulls"]
        if "mean_cost" in other and "cost_count" in other:
            # Weighted average of means
            w1 = self._cost_count
            w2 = other.get("cost_count", 0)
            if w1 + w2 > 0:
                self._mean_cost = (self._mean_cost * w1 + other["mean_cost"] * w2) / (w1 + w2)
            self._cost_count += other.get("cost_count", 0)
            self._cost_sum = self._mean_cost * self._cost_count
        if "arm_stats" in other:
            for nid, stats in other["arm_stats"].items():
                if nid in self._arm_stats:
                    self._arm_stats[nid]["sum_reward"] += stats["sum_reward"]
                    self._arm_stats[nid]["count"] += stats["count"]
                else:
                    self._arm_stats[nid] = dict(stats)


if __name__ == "__main__":
    from src.truesim import StochasticTrueModel

    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=15.0, memory_cap=15.0, network_cap=8.0),
    ]

    task = Task(
        id="t1",
        cpu_req=5.0,
        memory_req=5.0,
        network_req=2.0,
        arrival_time=0.0,
    )

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    mab = MeanFieldMABBalancer(seed=42)

    print("UCB1 bandit simulation (3 nodes, 10 tasks):")
    for i in range(10):
        chosen = mab.assign(task, nodes)
        cost = model.cost(task, chosen)
        mab.on_complete(task, chosen, cost)
        t = i + 1
        print(f"  task-{i}: -> {chosen.id}, cost={cost:.2f}, "
              f"total_pulls={t}, mean_cost={mab._mean_cost:.2f}")
        print(f"         UCBs: " + ", ".join(
            f"{n.id}={mab._arm_stats[n.id]['sum_reward']/max(mab._arm_stats[n.id]['count'],1):.2f}"
            f"+{math.sqrt(2*math.log(t)/max(mab._arm_stats[n.id]['count'],1)):.2f}"
            for n in nodes
        ))

    print(f"\nFinal snapshot: {mab.snapshot}")
