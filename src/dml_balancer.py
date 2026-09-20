"""Communication-Efficient DML Workload Balancer.

Based on: "Communication-Efficient Training Workload Balancing for
Decentralized Multi-Agent Learning" (arXiv:2405.00839).

Key ideas:
- Heterogeneity-aware balancing: nodes have different computation,
  communication, and storage capabilities.
- Tasks have different resource requirements; assigning to the right
  node minimizes straggler effects.
- Communication-efficient: the balancer makes decisions with limited
  information exchange, using locally-available state.

In our simulation framework, this maps to a weighted-load balancer where:
- Each node's "effective load" accounts for both queue depth and the
  resource demands of pending work relative to node capacity.
- Task-to-node fit considers how well the node's capability vector
  matches the task's resource requirements.
- The balancer prefers nodes that are both under-loaded AND capable
  of handling the task efficiently.

Two modes:
- 'queue_aware': weights queue depth by node capacity (simple, low comm)
- 'capability_aware': also considers task-specific resource fit (more precise)
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from src.node import Node
from src.task import Task


class CommunicationEfficientDMLBalancer:
    """Heterogeneity-aware workload balancer for decentralized multi-agent learning.

    Balances workload across heterogeneous nodes to minimize straggler effects.
    Uses weighted load scoring that accounts for both node capacity and
    task resource requirements.

    The core scoring formula for each node:
        score = load_weight / capacity_factor

    where:
        load_weight  = queue_depth * estimated_task_cost  (total pending work)
        capacity_factor = how fast this node processes this task type
                         (lower resource ratio = higher capacity_factor)

    Nodes with lower scores are preferred (less loaded relative to capability).
    """

    def __init__(
        self,
        seed: int = 42,
        mode: str = 'capability_aware',
        guessing_model = None,  # Optional GuessingModel for cost estimates
    ):
        """Parameters
        ----------
        seed: RNG seed (unused in deterministic scoring; kept for interface compat).
        mode: 'queue_aware' or 'capability_aware'.
            - queue_aware: score = queue_depth / geometric_mean_capacity
            - capability_aware: score = queue_depth * resource_ratio / capacity_factor
        guessing_model: optional GuessingModel for estimating task costs.
            If None, uses a simple resource-ratio heuristic.
        """
        self._seed = seed
        self._mode = mode
        self._guessing_model = guessing_model

        # Per-node observed cost history for learning
        self._node_cost_sums: dict[str, float] = {}
        self._node_cost_counts: dict[str, int] = {}
        self._node_avg_costs: dict[str, float] = {}

        # Assignment history for snapshot
        self._assignments: list[str] = []
        self._node_assign_counts: dict[str, int] = {}

    def _estimate_task_cost_on_node(
        self, task: Task, node: Node
    ) -> float:
        """Estimate how costly this task is on this node.

        Uses the bottleneck-ratio (max of resource ratios) as the cost estimate.
        The task's effective cost on a node is bounded by its most-constrained
        resource — this is the "weakest link" principle and matches how
        heterogeneity-aware balancing should work: a memory-bound task is
        expensive on a memory-poor node regardless of how much CPU it has.

        Falls back to arithmetic mean if bottleneck-ratio gives 0.
        """
        if self._guessing_model is not None:
            try:
                return self._guessing_model.estimate(task, node)
            except AttributeError:
                pass

        cpu_ratio = task.cpu_req / node.cpu_cap
        mem_ratio = task.memory_req / node.memory_cap
        net_ratio = task.network_req / node.network_cap
        # Bottleneck ratio: the task's cost is bounded by its worst-constrained resource
        bottleneck = max(cpu_ratio, mem_ratio, net_ratio)
        if bottleneck < 1e-9:
            # Fallback to arithmetic mean for degenerate cases
            return (cpu_ratio + mem_ratio + net_ratio) / 3.0
        return bottleneck


    def _estimated_queue_work(self, node: Node, task: Task) -> float:
        """Estimate the total work currently queued on this node.

        Approximates pending work as: queue_depth * estimated_cost_ofrepresentative_task.
        Uses the current task's estimated cost as a proxy for average pending work.
        """
        if node.queue_depth == 0:
            return 0.0
        # Use current task's estimated cost as a proxy for average pending work
        est_cost = self._estimate_task_cost_on_node(task, node)
        return node.queue_depth * est_cost

    def _node_score(
        self, task: Task, node: Node
    ) -> float:
        """Compute the score for assigning *task* to *node*.

        Lower score = better node.

        In queue_aware mode:
            score = queue_depth / geom_mean_capacity
            (Simple: prefers less-loaded nodes, scaled by capacity)

        In capability_aware mode:
            task_cost = bottleneck ratio (max of resource ratios) — how much
                this node struggles with this task's requirements
            queue_penalty = queue_depth * task_cost — existing load penalty
            score = task_cost + queue_penalty
            (Prefers nodes that can handle this task efficiently, with queue
             depth as a secondary penalty for already-loaded nodes)

        The bottleneck-ratio formula ensures that even empty nodes are ranked
        by how well they match the task's resource profile — the key insight
        from the DML paper.
        """
        if self._mode == 'queue_aware':
            geom_cap = math.sqrt(
                node.cpu_cap * node.memory_cap * node.network_cap
            )
            return node.queue_depth / max(geom_cap, 1e-9)
        else:  # capability_aware
            task_cost = self._estimate_task_cost_on_node(task, node)
            queue_penalty = self._estimated_queue_work(node, task)
            return task_cost + queue_penalty

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to the node with the best (lowest) weighted score.

        In queue_aware mode:  score = queue_depth / geom_mean_capacity
        In capability_aware mode: score = estimated_queue_work / capacity_factor
        """
        if not nodes:
            raise ValueError("No nodes available")

        best_node = None
        best_score = float('inf')

        for node in nodes:
            score = self._node_score(task, node)
            if score < best_score:
                best_score = score
                best_node = node

        assert best_node is not None

        # Track assignment
        self._assignments.append(best_node.id)
        self._node_assign_counts[best_node.id] = \
            self._node_assign_counts.get(best_node.id, 0) + 1

        return best_node

    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
        """Update per-node cost estimates from observed completion."""
        nid = node.id
        self._node_cost_sums[nid] = self._node_cost_sums.get(nid, 0.0) + actual_cost
        self._node_cost_counts[nid] = self._node_cost_counts.get(nid, 0) + 1
        count = self._node_cost_counts[nid]
        self._node_avg_costs[nid] = self._node_cost_sums[nid] / count

    @property
    def snapshot(self) -> dict:
        """Serializable state for gossip/decentralized exchange."""
        return {
            'mode': self._mode,
            'node_avg_costs': dict(self._node_avg_costs),
            'node_cost_counts': dict(self._node_cost_counts),
            'node_assign_counts': dict(self._node_assign_counts),
            'total_assignments': len(self._assignments),
            'assignments': self._assignments[-200:],
        }

    def merge(self, other: dict) -> None:
        """Merge another snapshot into this balancer's state."""
        if 'node_avg_costs' not in other or 'node_cost_counts' not in other:
            return

        for nid in other['node_avg_costs']:
            c1 = self._node_cost_counts.get(nid, 0)
            c2 = other['node_cost_counts'].get(nid, 0)
            if c1 + c2 > 0:
                self._node_avg_costs[nid] = (
                    self._node_avg_costs.get(nid, 0.0) * c1
                    + other['node_avg_costs'][nid] * c2
                ) / (c1 + c2)
                self._node_cost_counts[nid] = c1 + c2
                self._node_cost_sums[nid] = self._node_avg_costs[nid] * self._node_cost_counts[nid]

        if 'node_assign_counts' in other:
            for nid, count in other['node_assign_counts'].items():
                self._node_assign_counts[nid] = \
                    self._node_assign_counts.get(nid, 0) + count

        if 'total_assignments' in other:
            self._assignments.extend(['merged'] * other['total_assignments'])


if __name__ == "__main__":
    nodes = [
        Node(id="n0", cpu_cap=30.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=25.0, network_cap=8.0),
        Node(id="n2", cpu_cap=20.0, memory_cap=15.0, network_cap=15.0),
    ]

    # Pre-load some tasks on each node to simulate existing load
    nodes[0].queue = ["t-0", "t-1", "t-2"]  # n0: 3 tasks
    nodes[1].queue = ["t-3"]  # n1: 1 task
    nodes[2].queue = ["t-4", "t-5"]  # n2: 2 tasks

    task = Task(
        id="new-task",
        cpu_req=8.0,
        memory_req=12.0,
        network_req=5.0,
        arrival_time=0.0,
    )

    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')

    print("Communication-Efficient DML Balancer demo:")
    print(f"  Node capacities: n0(cpu=30,mem=20,net=10) n1(cpu=10,mem=25,net=8) n2(cpu=20,mem=15,net=15)")
    print(f"  Node queue depths: n0={nodes[0].queue_depth} n1={nodes[1].queue_depth} n2={nodes[2].queue_depth}")
    print(f"  Task requirements: cpu={task.cpu_req} mem={task.memory_req} net={task.network_req}")
    print()

    for mode in ['queue_aware', 'capability_aware']:
        balancer._mode = mode
        chosen = balancer.assign(task, nodes)
        print(f"  mode={mode}: chose {chosen.id}")

        # Show scores
        for node in nodes:
            if mode == 'queue_aware':
                geom_cap = math.sqrt(node.cpu_cap * node.memory_cap * node.network_cap)
                score = node.queue_depth / max(geom_cap, 1e-9)
                print(f"    {node.id}: score={score:.4f} (queue_depth={node.queue_depth}, geom_cap={geom_cap:.1f})")
            else:
                task_cost = balancer._estimate_task_cost_on_node(task, node)
                queue_work = balancer._estimated_queue_work(node, task)
                score = task_cost + queue_work
                print(f"    {node.id}: score={score:.4f} (task_cost={task_cost:.3f}, queue_work={queue_work:.3f})")

    print()
    print(f"Snapshot: {balancer.snapshot}")
