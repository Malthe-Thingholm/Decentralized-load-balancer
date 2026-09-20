"""RL-based adaptive load balancer using tabular Q-learning.

Based on: "Reinforcement Learning-Based Adaptive Load Balancing for
Dynamic Cloud Environments" (arXiv:2409.04896).

Key ideas:
- An RL agent learns to assign tasks to nodes by observing system state.
- State: discretized node loads (queue depths relative to capacity) +
  recent average costs per node.
- Action: which node to assign the current task to.
- Reward: negative of the observed cost (lower cost = higher reward).
- Q-learning update: Q(s,a) += alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
- Epsilon-greedy exploration: with prob epsilon, pick a random node; else
  pick the node with highest Q-value for the current state.

This implementation uses a simplified state representation:
- Discretize each node's queue depth into bins (0, 1, 2, 3+).
- Discretize each node's average cost into bins (low, medium, high).
- State = tuple of (queue_bin_i, cost_bin_i) for all nodes.
- Q-table: dict mapping (state_tuple, action) -> Q-value.

The agent learns online during the simulation — no pre-training required.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from src.node import Node
from src.task import Task


class QLearningBalancer:
    """Tabular Q-learning load balancer.

    Learns an assignment policy online by observing task costs.
    Uses epsilon-greedy exploration and Q-learning updates.

    Attributes:
        q_table: dict mapping (state_tuple, action_idx) -> float Q-value.
        state_history: list of (state, action, reward) tuples for analysis.
    """

    def __init__(
        self,
        seed: int = 42,
        num_nodes: int | None = None,
        alpha: float = 0.1,  # learning rate
        gamma: float = 0.9,  # discount factor
        epsilon: float = 0.3,  # exploration probability
        epsilon_decay: float = 0.995,  # per-step decay
        epsilon_min: float = 0.05,
        queue_bins: int = 4,  # number of discretization bins for queue depth
        cost_bins: int = 3,  # number of discretization bins for avg cost
        max_cost_scale: float = 20.0,  # costs above this are capped for binning
    ):
        """Parameters
        ----------
        seed: RNG seed for exploration.
        num_nodes: pre-allocate if known (needed for Q-table shape).
        alpha: learning rate (0 = no learning, 1 = full replacement).
        gamma: discount factor for future rewards.
        epsilon: initial exploration probability.
        epsilon_decay: multiplicative decay per step.
        epsilon_min: minimum exploration probability.
        queue_bins: discretization levels for queue depth.
        cost_bins: discretization levels for average cost.
        max_cost_scale: costs above this are capped to this value for binning.
        """
        self._rng = random.Random(seed)
        self._num_nodes = num_nodes
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._queue_bins = queue_bins
        self._cost_bins = cost_bins
        self._max_cost_scale = max_cost_scale

        # Q-table: (state_tuple, action) -> Q-value
        self.q_table: dict[tuple, float] = {}

        # Per-node running statistics (for state computation)
        self._queue_depths: dict[str, int] = {}  # node_id -> current queue depth
        self._cost_sums: dict[str, float] = {}  # node_id -> sum of costs
        self._cost_counts: dict[str, int] = {}  # node_id -> count
        self._avg_costs: dict[str, float] = {}  # node_id -> running avg

        # Exploration tracking
        self._total_steps = 0
        self._current_epsilon = epsilon

        # History
        self._state_history: list[tuple] = []  # (state, action, reward)
        self._assignments: list[str] = []
        self._actions_taken: list[int] = []

    def _discretize_queue(self, depth: int) -> int:
        """Discretize queue depth into bins: 0, 1, 2, ..., queue_bins-1."""
        return min(depth, self._queue_bins - 1)

    def _discretize_cost(self, avg_cost: float) -> int:
        """Discretize average cost into bins: 0 (low), 1 (medium), 2 (high)."""
        capped = min(avg_cost, self._max_cost_scale)
        # Divide into cost_bins equal ranges
        bin_size = self._max_cost_scale / self._cost_bins
        return min(int(capped / bin_size), self._cost_bins - 1)

    def _get_state(self, nodes: list[Node], snapshot_queue_depths: bool = False) -> tuple:
        """Compute the current state representation.

        State = tuple of (queue_bin_i, cost_bin_i) for all nodes,
        ordered by node id.

        If snapshot_queue_depths is True, use the stored queue depths
        (pre-assignment snapshot) instead of live node.queue_depth.
        """
        # Sort nodes by id for consistent state ordering
        sorted_nodes = sorted(nodes, key=lambda n: n.id)

        state_parts = []
        for node in sorted_nodes:
            if snapshot_queue_depths:
                q = self._queue_depths[node.id]
            else:
                q = node.queue_depth
                self._queue_depths[node.id] = q
            q_bin = self._discretize_queue(q)
            c_bin = self._discretize_cost(self._avg_costs.get(node.id, 0.0))
            state_parts.append(q_bin)
            state_parts.append(c_bin)

        return tuple(state_parts)

    def _get_q(self, state: tuple, action: int) -> float:
        """Get Q-value for (state, action), defaulting to 0.0."""
        key = (state, action)
        return self.q_table.get(key, 0.0)

    def _set_q(self, state: tuple, action: int, value: float) -> None:
        """Set Q-value for (state, action)."""
        self.q_table[(state, action)] = value

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* using epsilon-greedy Q-learning policy."""
        if not nodes:
            raise ValueError("No nodes available")

        # Ensure per-node tracking exists
        for node in nodes:
            if node.id not in self._queue_depths:
                self._queue_depths[node.id] = 0
            if node.id not in self._cost_sums:
                self._cost_sums[node.id] = 0.0
                self._cost_counts[node.id] = 0
                self._avg_costs[node.id] = 0.0

        # Capture state BEFORE updating queue depths (pre-assignment state)
        state = self._get_state(nodes, snapshot_queue_depths=True)

        n = len(nodes)

        # Epsilon-greedy action selection
        if self._rng.random() < self._current_epsilon:
            # Explore: random node
            action = self._rng.randint(0, n - 1)
        else:
            # Exploit: pick node with highest Q-value for this state
            best_action = 0
            best_q = self._get_q(state, 0)
            for a in range(1, n):
                q = self._get_q(state, a)
                if q > best_q:
                    best_q = q
                    best_action = a
            action = best_action

        chosen = nodes[action]
        self._assignments.append(chosen.id)
        self._actions_taken.append(action)
        self._total_steps += 1

        # Decay epsilon
        self._current_epsilon = max(
            self._epsilon_min,
            self._current_epsilon * self._epsilon_decay,
        )

        # Store for Q-update after observing reward
        self._last_state = state
        self._last_action = action

        return chosen

    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
        """Update Q-table with observed reward (negative cost)."""
        nid = node.id
        if nid not in self._cost_sums:
            return

        # Update running average cost
        cnt = self._cost_counts[nid] + 1
        self._cost_counts[nid] = cnt
        self._cost_sums[nid] += actual_cost
        self._avg_costs[nid] = self._cost_sums[nid] / cnt

        # Reward = negative normalized cost (higher = better)
        reward = -actual_cost / self._max_cost_scale

        # Q-learning update
        if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
            state = self._last_state
            action = self._last_action

            # Current Q-value
            q_current = self._get_q(state, action)

            # Max Q-value for next state (we don't have next state here —
            # use 0 as approximation for terminal step, or estimate)
            # For online learning without explicit next state, we use:
            # Q(s,a) += alpha * (reward - Q(s,a))
            # This is TD(0) with no future estimate, which is valid for
            # episodic tasks where the episode ends after each assignment.
            td_target = reward  # no future reward estimate
            td_error = td_target - q_current
            new_q = q_current + self._alpha * td_error
            self._set_q(state, action, new_q)

            self._state_history.append((state, action, reward))

    @property
    def snapshot(self) -> dict:
        """Serializable state for analysis."""
        return {
            "q_table_size": len(self.q_table),
            "total_steps": self._total_steps,
            "current_epsilon": self._current_epsilon,
            "avg_costs": dict(self._avg_costs),
            "cost_counts": dict(self._cost_counts),
            "assignments": self._assignments[-100:],
            "actions_taken": self._actions_taken[-100:],
            "state_history_len": len(self._state_history),
        }

    def merge(self, other: dict) -> None:
        """Merge another snapshot into this balancer's state.

        Merges Q-tables by averaging values for overlapping (state, action)
        pairs, and combines per-node statistics.
        """
        if "avg_costs" not in other or "cost_counts" not in other:
            return

        for nid in other["avg_costs"]:
            if nid not in self._avg_costs:
                continue
            c1 = self._cost_counts.get(nid, 0)
            c2 = other["cost_counts"].get(nid, 0)
            if c1 + c2 > 0:
                self._avg_costs[nid] = (
                    self._avg_costs[nid] * c1 + other["avg_costs"][nid] * c2
                ) / (c1 + c2)
                self._cost_counts[nid] = c1 + c2
                self._cost_sums[nid] = self._avg_costs[nid] * self._cost_counts[nid]

        # Merge Q-tables (simple averaging for overlapping entries)
        # Note: other may not have q_table key if it's a different snapshot type
        if "q_table" in other and isinstance(other["q_table"], dict):
            for key, q_val in other["q_table"].items():
                if key in self.q_table:
                    # Weighted by visit count — approximate with simple average
                    self.q_table[key] = (self.q_table[key] + q_val) / 2
                else:
                    self.q_table[key] = q_val


if __name__ == "__main__":
    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=15.0, memory_cap=15.0, network_cap=8.0),
    ]

    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0)

    from src.truesim import StochasticTrueModel
    model = StochasticTrueModel(seed=42, cost_scale=15.0)

    rl = QLearningBalancer(seed=42, alpha=0.2, gamma=0.9, epsilon=0.5,
                           epsilon_decay=0.99, epsilon_min=0.1)

    print("RL Q-learning balancer (3 nodes, 30 tasks):")
    for i in range(30):
        chosen = rl.assign(task, nodes)
        cost = model.cost(task, chosen)
        rl.on_complete(task, chosen, cost)
        snap = rl.snapshot
        print(f"  task-{i}: -> {chosen.id}, cost={cost:.2f}, "
              f"eps={snap['current_epsilon']:.3f}, "
              f"q_size={snap['q_table_size']}, "
              f"avg_costs={ {k: f'{v:.2f}' for k, v in snap['avg_costs'].items()} }")

    print(f"\nFinal snapshot: {rl.snapshot}")
