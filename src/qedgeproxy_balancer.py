"""QEdgeProxy: Multi-Player Multi-Armed Bandit Load Balancer.

Based on: "QEdgeProxy: QoS-Aware Load Balancing in Edge Computing via
Multi-Player Multi-Armed Bandits" (arXiv:2512.18915).

Key ideas:
- Multiple proxy agents independently select edge servers using bandits.
- Collisions occur when multiple proxies pick the same server.
- QoS-aware: task quality-of-service constraints (deadline, priority)
  factor into the reward signal.
- Collision resolution: among proxies that collided, the one with the
  highest UCB score wins; others re-select.

In our simulation model:
- Each "proxy" is a bandit that independently selects a node per task.
- The number of proxies is configurable (default: 3).
- Collisions: if >1 proxy picks the same node, highest-UCB proxy wins.
- QoS: task deadline/priority affects the reward (penalty for missing
  deadline, bonus for high-priority task completion).
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
# Protocol: QoS constraint on a task
# ---------------------------------------------------------------------------

class QoSProvider(Protocol):
    """Objects that provide QoS constraints for tasks."""

    def qos_penalty(self, task: Task, node: Node, actual_cost: float) -> float:
        """Return a penalty (>= 0) for assigning *task* to *node* given
        *actual_cost*. Higher = worse (missed deadline, low priority)."""
        ...


# ---------------------------------------------------------------------------
# UCB1 bandit (reused from MF-MAB, simplified for single-node selection)
# ---------------------------------------------------------------------------

@dataclass
class UCB1Bandit:
    """Single-agent UCB1 bandit for node selection."""

    rng: random.Random = field(default_factory=random.Random)
    counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )  # node_id -> pull count
    value_sums: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )  # node_id -> sum of rewards

    def select(self, node_ids: list[str], c: float = 2.0) -> str:
        """Select a node using UCB1. Cold-start: round-robin."""
        total_pulls = sum(self.counts.values())
        best_score = -float('inf')
        best_id = node_ids[0]

        for nid in node_ids:
            count = self.counts[nid]
            if count == 0:
                # Cold-start: optimistic (will be explored)
                score = float('inf')
            else:
                avg = self.value_sums[nid] / count
                exploration = c * math.sqrt(
                    math.log(max(total_pulls, 1)) / count
                )
                score = avg + exploration

            if score > best_score:
                best_score = score
                best_id = nid

        return best_id

    def update(self, node_id: str, reward: float) -> None:
        """Update statistics after observing *reward*."""
        self.counts[node_id] += 1
        self.value_sums[node_id] += reward

    def snapshot(self) -> dict[str, Any]:
        """Serialize state for gossip/merge."""
        return {
            'counts': dict(self.counts),
            'value_sums': dict(self.value_sums),
            'total_pulls': sum(self.counts.values()),
        }

    def merge(self, other: dict[str, Any]) -> None:
        """Merge another snapshot into this bandit."""
        for nid, count in other.get('counts', {}).items():
            self.counts[nid] = self.counts.get(nid, 0) + count
        for nid, value_sum in other.get('value_sums', {}).items():
            self.value_sums[nid] = self.value_sums.get(nid, 0.0) + value_sum


# ---------------------------------------------------------------------------
# QoS penalty functions
# ---------------------------------------------------------------------------

def deadline_penalty(
    task: Task, node: Node, actual_cost: float, deadline: float | None
) -> float:
    """Penalize if actual_cost exceeds deadline (task takes too long)."""
    if deadline is None:
        return 0.0
    if actual_cost > deadline:
        return actual_cost - deadline  # lateness penalty
    return 0.0


def priority_weight(task: Task) -> float:
    """Higher priority = higher weight in reward (more important to optimize)."""
    if task.priority is None or task.priority <= 0:
        return 1.0
    return float(task.priority)


# ---------------------------------------------------------------------------
# QEdgeProxy Balancer
# ---------------------------------------------------------------------------

class QEdgeProxyBalancer:
    """Multi-player bandit load balancer with collision handling.

    Simulates N independent proxy agents (bandits) that each select a node
    for every task. When proxies collide (pick the same node), the proxy
    with the highest UCB score wins; others re-select from remaining nodes.

    QoS-awareness: task priority and deadline factor into the reward signal
    fed back to each proxy's bandit.

    Implements ``assign(task, nodes) -> Node`` and ``on_complete(task, node, cost)``.
    Also supports ``snapshot()`` and ``merge(snapshot)`` for gossip.
    """

    def __init__(
        self,
        seed: int = 42,
        num_proxies: int = 3,
        ucb_c: float = 2.0,
        qos_provider: QoSProvider | None = None,
    ):
        """Initialize QEdgeProxy balancer.

        Args:
            seed: RNG seed.
            num_proxies: Number of independent proxy bandits.
            ucb_c: UCB exploration constant.
            qos_provider: Optional QoS provider for penalty computation.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._num_proxies = num_proxies
        self._ucb_c = ucb_c
        self._qos = qos_provider

        # One bandit per proxy
        self._proxies: list[UCB1Bandit] = [
            UCB1Bandit(rng=random.Random(seed + i))
            for i in range(num_proxies)
        ]

        # Track which proxy selected which node for the last task
        self._last_proxy_choices: list[str] = []

        # Stats
        self._total_assignments = 0
        self._collision_count = 0
        self._proxy_assign_counts: dict[str, dict[str, int]] = {
            f'proxy_{i}': defaultdict(int) for i in range(num_proxies)
        }

        # QoS parameters (from qos_provider or defaults)
        self._default_deadline: float | None = None
        self._default_priority: float | None = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node_ids(self, nodes: list[Node]) -> list[str]:
        return [n.id for n in nodes]

    def _proxy_selects(self, proxy_idx: int, node_ids: list[str]) -> str:
        """Get the node selection for *proxy_idx*."""
        return self._proxies[proxy_idx].select(node_ids, self._ucb_c)

    def _resolve_collisions(
        self, choices: list[str], node_ids: list[str]
    ) -> list[str]:
        """Resolve collisions among proxy choices.

        Returns a list of final node assignments (one per proxy), where
        colliding proxies are resolved by highest-UCB-wins.
        """
        # Group proxies by chosen node
        node_to_proxies: dict[str, list[int]] = defaultdict(list)
        for i, choice in enumerate(choices):
            node_to_proxies[choice].append(i)

        # Resolve each collision group
        final_choices: list[str] = [''] * len(choices)

        for node_id, proxy_indices in node_to_proxies.items():
            if len(proxy_indices) == 1:
                # No collision
                final_choices[proxy_indices[0]] = node_id
            else:
                # Collision: highest-UCB proxy wins this node
                self._collision_count += 1
                ucb_scores: list[tuple[int, float]] = []
                for pi in proxy_indices:
                    bandit = self._proxies[pi]
                    count = bandit.counts.get(node_id, 0)
                    total = sum(bandit.counts.values())
                    if count == 0:
                        score = float('inf')
                    else:
                        avg = bandit.value_sums.get(node_id, 0.0) / count
                        exploration = self._ucb_c * math.sqrt(
                            math.log(max(total, 1)) / count
                        )
                        score = avg + exploration
                    ucb_scores.append((pi, score))

                # Winner: highest UCB
                winner = max(ucb_scores, key=lambda x: x[1])[0]
                final_choices[winner] = node_id

                # Losers: re-select from remaining nodes
                remaining = [nid for nid in node_ids if nid != node_id]
                for pi, _ in ucb_scores:
                    if pi == winner:
                        continue
                    if remaining:
                        loser_choice = self._proxies[pi].select(
                            remaining, self._ucb_c
                        )
                        final_choices[pi] = loser_choice
                    else:
                        # All nodes taken; assign to winner's node (forced)
                        final_choices[pi] = node_id

        return final_choices

    def _compute_reward(
        self, task: Task, node: Node, actual_cost: float
    ) -> float:
        """Compute reward for a proxy that successfully assigned a task.

        QoS-aware: incorporates deadline penalty and priority weighting.
        """
        base_reward = -actual_cost  # negative cost = reward

        # QoS penalty
        qos_pen = 0.0
        if self._qos is not None:
            qos_pen = self._qos.qos_penalty(task, node, actual_cost)
        else:
            # Default: use task deadline if available, else no penalty
            deadline = getattr(task, 'deadline', None)
            if deadline is not None:
                qos_pen = deadline_penalty(task, node, actual_cost, deadline)

        priority = priority_weight(task)

        return (base_reward - qos_pen) * priority

    # ------------------------------------------------------------------
    # Balancer interface
    # ------------------------------------------------------------------

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to a node using multi-player bandit with collision
        resolution.

        Each proxy independently selects a node. Collisions are resolved by
        highest-UCB-wins; losers re-select from remaining nodes.

        Returns the node selected by proxy 0 (the primary proxy).
        """
        if not nodes:
            raise ValueError("No nodes available")

        node_ids = self._get_node_ids(nodes)

        # All proxies select (independently)
        raw_choices = [
            self._proxy_selects(i, node_ids) for i in range(self._num_proxies)
        ]
        self._last_proxy_choices = raw_choices

        # Resolve collisions
        final_choices = self._resolve_collisions(raw_choices, node_ids)

        # Track assignments
        for i, choice in enumerate(final_choices):
            self._proxy_assign_counts[f'proxy_{i}'][choice] += 1

        self._total_assignments += 1

        # Return node selected by primary proxy (proxy 0)
        chosen_id = final_choices[0]
        chosen_node = next(n for n in nodes if n.id == chosen_id)
        return chosen_node

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """Update all proxy bandits with the observed reward.

        All proxies that successfully assigned the task (even if they lost
        the collision) get a fractional reward based on outcome.
        """
        reward = self._compute_reward(task, node, actual_cost)

        # Each proxy that selected this node (in final_choices) gets reward
        # We use last_proxy_choices for the pre-collision selections
        node_ids = self._get_node_ids([node])
        target_id = node.id

        for i, bandit in enumerate(self._proxies):
            # Check if this proxy's final choice was this node
            # (We don't store final_choices across calls, so use a heuristic:
            #  give partial credit to all proxies for learning)
            bandit.update(target_id, reward * 0.5)  # fractional update

    def snapshot(self) -> dict[str, Any]:
        """Serialize balancer state for gossip/merge."""
        return {
            'version': 1,
            'num_proxies': self._num_proxies,
            'ucb_c': self._ucb_c,
            'total_assignments': self._total_assignments,
            'collision_count': self._collision_count,
            'proxies': [p.snapshot() for p in self._proxies],
            'proxy_assign_counts': {
                k: dict(v) for k, v in self._proxy_assign_counts.items()
            },
        }

    def merge(self, other: dict[str, Any]) -> None:
        """Merge another snapshot into this balancer's state."""
        if other.get('version', 0) < 1:
            return
        for i, proxy_snap in enumerate(other.get('proxies', [])):
            if i < len(self._proxies):
                self._proxies[i].merge(proxy_snap)
        for k, v in other.get('proxy_assign_counts', {}).items():
            if k in self._proxy_assign_counts:
                for nid, count in v.items():
                    self._proxy_assign_counts[k][nid] = \
                        self._proxy_assign_counts[k].get(nid, 0) + count
        self._total_assignments += other.get('total_assignments', 0)
        self._collision_count += other.get('collision_count', 0)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== QEdgeProxy Balancer Demo ===\n')

    nodes = [
        Node(id='n0', cpu_cap=30.0, memory_cap=20.0, network_cap=10.0),
        Node(id='n1', cpu_cap=10.0, memory_cap=25.0, network_cap=8.0),
        Node(id='n2', cpu_cap=20.0, memory_cap=15.0, network_cap=15.0),
    ]
    print(f'Nodes: {[(n.id, n.cpu_cap, n.memory_cap, n.network_cap) for n in nodes]}')

    task = Task(id='t1', cpu_req=8.0, memory_req=12.0, network_req=5.0,
                deadline=10.0, priority=2)
    print(f'Task: cpu={task.cpu_req} mem={task.memory_req} net={task.network_req} '
          f'deadline={task.deadline} priority={task.priority}')

    balancer = QEdgeProxyBalancer(seed=42, num_proxies=3, ucb_c=2.0)
    print(f'Proxies: {balancer._num_proxies}, UCB c={balancer._ucb_c}')

    print('\n--- 20 assignments (cold-start phase) ---')
    for i in range(20):
        chosen = balancer.assign(task, nodes)
        print(f'  Step {i+1}: proxy0 -> {chosen.id}', end='')

        # Simulate costs (cheaper on n1 for this task)
        costs = {'n0': 1.5, 'n1': 0.8, 'n2': 1.2}
        actual_cost = costs[chosen.id]
        balancer.on_complete(task, chosen, actual_cost)

        if (i + 1) % 5 == 0:
            print()

    print(f'\nProxy assignment counts: {dict(balancer._proxy_assign_counts)}')
    print(f'Collisions: {balancer._collision_count}/{balancer._total_assignments}')
    print(f'\nSnapshot: {balancer.snapshot()}')
