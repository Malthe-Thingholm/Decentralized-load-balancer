"""Incentive-Based Decentralized Load Balancer.

Based on: "Model of an Open, Decentralized Computational Network with
Incentive-Based Load Balancing" (arXiv:2501.01219).

Key ideas:
- Nodes earn reputation from successful task completions.
- Reputation affects node "attractiveness" — high-rep nodes get more work.
- Tasks can prefer high-rep nodes (quality assurance) or low-rep nodes
  (cost savings, since low-rep nodes may offer lower effective prices).
- The balancer uses a reputation-weighted cost: effective_cost = raw_cost /
  reputation (higher rep = lower effective cost = more attractive).
- Reputation decays over time (forgetting factor) to adapt to changing
  node performance.

In our simulation model:
- Each node has a reputation score in [0.1, 1.0] (floor prevents division
  by zero and extreme punishment).
- Reputation updated on task completion: increases if actual_cost <
  expected_cost, decreases otherwise.
- Balancer assigns to node with lowest reputation-weighted cost.
- Supports optional QoS/priority weighting on top of reputation.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.balancer import Balancer
from src.node import Node
from src.task import Task


# ---------------------------------------------------------------------------
# Protocol: reputation-updatable nodes
# ---------------------------------------------------------------------------

class ReputationNode(Protocol):
    """A node that carries a reputation score."""

    id: str
    reputation: float  # in [0.1, 1.0]


# ---------------------------------------------------------------------------
# Incentive Balancer
# ---------------------------------------------------------------------------

@dataclass
class IncentiveBalancer:
    """Reputation and pricing-based decentralized load balancer.

    Assigns tasks to nodes with the lowest reputation-weighted cost.
    Reputation is updated on task completion and decays over time.

    Two modes:
    - 'reputation_weighted': effective_cost = raw_cost / reputation
      (high-rep nodes are more attractive)
    - 'price_aware': nodes have explicit prices; balancer minimizes
      price * raw_cost, with reputation as a tiebreaker

    Also supports snapshot/merge for gossip.
    """

    seed: int = 42
    mode: str = 'reputation_weighted'
    reputation_floor: float = 0.1
    reputation_cap: float = 1.0
    reputation_update_rate: float = 0.1
    reputation_decay: float = 0.001  # per-step decay toward neutral (0.5)
    neutral_reputation: float = 0.5
    node_prices: dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 1.0)
    )  # node_id -> price multiplier
    rng: random.Random = field(default_factory=random.Random)

    # Internal state
    _reputations: dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 0.5)
    )  # node_id -> reputation
    _expected_costs: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )  # node_id -> running avg of observed costs
    _cost_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _total_assignments: int = 0
    _node_assign_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def _effective_cost(
        self, task: Task, node: Node, raw_cost: float
    ) -> float:
        """Compute the effective (reputation-weighted) cost for assigning
        *task* to *node* given *raw_cost*."""
        if self.mode == 'reputation_weighted':
            rep = self._get_reputation(node.id)
            return raw_cost / max(rep, self.reputation_floor)
        elif self.mode == 'price_aware':
            price = self.node_prices.get(node.id, 1.0)
            rep = self._get_reputation(node.id)
            # Price * cost, with reputation as tiebreaker (lower rep = slight penalty)
            return price * raw_cost / max(rep, self.reputation_floor)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_reputation(self, node_id: str) -> float:
        """Get current reputation for *node_id*."""
        return self._reputations.get(node_id, self.neutral_reputation)

    def _update_reputation(
        self, node_id: str, actual_cost: float, expected_cost: float
    ) -> None:
        """Update reputation for *node_id* based on observed vs expected cost.

        Reputation increases if actual < expected (node performed better
        than expected), decreases otherwise.
        """
        current = self._get_reputation(node_id)
        delta = self.reputation_update_rate * (
            expected_cost - actual_cost
        ) / max(expected_cost, 1e-9)
        new_rep = current + delta
        new_rep = max(self.reputation_floor, min(self.reputation_cap, new_rep))
        self._reputations[node_id] = new_rep

    def _update_expected_cost(self, node_id: str, actual_cost: float) -> None:
        """Update running average of observed costs for *node_id*."""
        count = self._cost_counts.get(node_id, 0)
        if count == 0:
            self._expected_costs[node_id] = actual_cost
        else:
            old_avg = self._expected_costs[node_id]
            self._expected_costs[node_id] = (
                old_avg * count + actual_cost
            ) / (count + 1)
        self._cost_counts[node_id] = count + 1

    def _apply_decay(self) -> None:
        """Apply reputation decay toward neutral."""
        for nid in list(self._reputations.keys()):
            current = self._reputations[nid]
            # Decay toward neutral
            self._reputations[nid] = current + self.reputation_decay * (
                self.neutral_reputation - current
            )

    # ------------------------------------------------------------------
    # Balancer interface
    # ------------------------------------------------------------------

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to the node with the lowest reputation-weighted cost.

        Uses estimated cost (from _expected_costs or a heuristic) combined
        with reputation to score each node.
        """
        if not nodes:
            raise ValueError("No nodes available")

        self._apply_decay()

        best_node = None
        best_score = float('inf')

        for node in nodes:
            # Estimate cost for this task on this node
            if self._cost_counts.get(node.id, 0) > 0:
                raw_cost = self._expected_costs[node.id]
            else:
                # Cold-start: use a heuristic based on node capacity
                # (lower capacity → higher expected cost)
                geom_cap = math.sqrt(
                    node.cpu_cap * node.memory_cap * node.network_cap
                )
                raw_cost = 50.0 / max(geom_cap, 1e-9)

            score = self._effective_cost(task, node, raw_cost)

            if score < best_score:
                best_score = score
                best_node = node

        assert best_node is not None

        self._node_assign_counts[best_node.id] = \
            self._node_assign_counts.get(best_node.id, 0) + 1
        self._total_assignments += 1

        return best_node

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """Update reputation and expected cost for *node* based on
        observed *actual_cost*."""
        nid = node.id
        if self._cost_counts.get(nid, 0) == 0:
            # Cold-start: use capacity-based heuristic as expected cost
            geom_cap = math.sqrt(
                node.cpu_cap * node.memory_cap * node.network_cap
            )
            expected = 50.0 / max(geom_cap, 1e-9)
        else:
            expected = self._expected_costs[nid]
        self._update_reputation(nid, actual_cost, expected)
        self._update_expected_cost(nid, actual_cost)

    def snapshot(self) -> dict[str, Any]:
        """Serialize balancer state for gossip/merge."""
        return {
            'version': 1,
            'mode': self.mode,
            'reputation_floor': self.reputation_floor,
            'reputation_cap': self.reputation_cap,
            'reputation_update_rate': self.reputation_update_rate,
            'reputation_decay': self.reputation_decay,
            'neutral_reputation': self.neutral_reputation,
            'node_prices': dict(self.node_prices),
            'reputations': dict(self._reputations),
            'expected_costs': dict(self._expected_costs),
            'cost_counts': dict(self._cost_counts),
            'total_assignments': self._total_assignments,
            'node_assign_counts': dict(self._node_assign_counts),
        }

    def merge(self, other: dict[str, Any]) -> None:
        """Merge another snapshot into this balancer's state."""
        if other.get('version', 0) < 1:
            return
        # Merge reputations (weighted average by observation count)
        for nid, other_rep in other.get('reputations', {}).items():
            my_count = self._cost_counts.get(nid, 0)
            other_count = other.get('cost_counts', {}).get(nid, 0)
            if my_count + other_count > 0:
                total = my_count + other_count
                my_rep = self._reputations.get(nid, self.neutral_reputation)
                self._reputations[nid] = (
                    my_rep * my_count + other_rep * other_count
                ) / total
            else:
                self._reputations[nid] = other_rep

        # Merge expected costs
        for nid, other_cost in other.get('expected_costs', {}).items():
            my_count = self._cost_counts.get(nid, 0)
            other_count = other.get('cost_counts', {}).get(nid, 0)
            if my_count + other_count > 0:
                total = my_count + other_count
                my_cost = self._expected_costs.get(nid, 0.0)
                self._expected_costs[nid] = (
                    my_cost * my_count + other_cost * other_count
                ) / total
            else:
                self._expected_costs[nid] = other_cost

        # Merge counts
        for nid, count in other.get('cost_counts', {}).items():
            self._cost_counts[nid] = self._cost_counts.get(nid, 0) + count

        # Merge assign counts
        for nid, count in other.get('node_assign_counts', {}).items():
            self._node_assign_counts[nid] = \
                self._node_assign_counts.get(nid, 0) + count

        self._total_assignments += other.get('total_assignments', 0)

        # Merge node prices
        for nid, price in other.get('node_prices', {}).items():
            self.node_prices[nid] = price


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Incentive-Based Load Balancer Demo ===\n')

    nodes = [
        Node(id='n0', cpu_cap=30.0, memory_cap=20.0, network_cap=10.0),
        Node(id='n1', cpu_cap=10.0, memory_cap=25.0, network_cap=8.0),
        Node(id='n2', cpu_cap=20.0, memory_cap=15.0, network_cap=15.0),
    ]
    print(f'Nodes: {[(n.id, n.cpu_cap, n.memory_cap, n.network_cap) for n in nodes]}')

    task = Task(id='t1', cpu_req=8.0, memory_req=12.0, network_req=5.0,
                deadline=10.0, priority=1)
    print(f'Task: cpu={task.cpu_req} mem={task.memory_req} net={task.network_req}')

    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    print(f'Mode: {balancer.mode}')

    print('\n--- 20 assignments ---')
    for i in range(20):
        chosen = balancer.assign(task, nodes)
        print(f'  Step {i+1}: chose {chosen.id}', end='')

        # Simulate: n1 is cheapest for this task
        costs = {'n0': 2.0, 'n1': 1.0, 'n2': 1.5}
        actual_cost = costs[chosen.id]
        balancer.on_complete(task, chosen, actual_cost)

        if (i + 1) % 5 == 0:
            print()

    print(f'\nNode reputations: {dict(balancer._reputations)}')
    print(f'Node assign counts: {dict(balancer._node_assign_counts)}')
    print(f'\nSnapshot: {balancer.snapshot()}')
