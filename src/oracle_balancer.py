"""Oracle (Hungarian) balancer — computes optimal assignment of tasks to nodes.

Solves |Tasks| × |Nodes| assignment to minimize total cost.
This is the ground-truth benchmark. In practice, Hungarian is O(n³)
so it's expensive for large sims — used as reference only.
"""

from __future__ import annotations

from src.node_descriptors import Node
from src.task_descriptors import Task
from src.true_estimates import TrueEstimator


class OracleBalancer:
    """Optimal assignment via brute-force enumeration for small instances.
    Ground-truth benchmark for regret comparison.
    """

    def __init__(self, true_estimator: TrueEstimator | None = None):
        self._estimator = true_estimator

    def assign(
        self, task: Task, nodes: list[Node]
    ) -> Node:
        # Single-task version: just pick cheapest node
        if self._estimator is not None:
            costs = [self._estimator.estimate(task, n) for n in nodes]
            best_idx = min(range(len(costs)), key=lambda i: costs[i])
            return nodes[best_idx]
        return min(nodes, key=lambda n: n.capacity)

    def batch_assign(
        self,
        tasks: list[Task],
        nodes: list[Node],
        estimator: TrueEstimator,
    ) -> list[tuple[Task, Node, float]]:
        """Brute-force optimal batch assignment (for small n).
        Returns list of (task, node, cost) tuples.
        """
        n_tasks = len(tasks)
        n_nodes = len(nodes)

        if n_tasks == 0 or n_nodes == 0:
            return []

        # For small instances: enumerate all possible assignments
        # Each task assigned to one node, tasks are distinguishable
        best_cost = float('inf')
        best_assignment: list[tuple[Task, Node, float]] = []

        def try_assignment(task_idx: int, current: list[tuple[Task, Node, float]]) -> None:
            nonlocal best_cost, best_assignment
            if task_idx == n_tasks:
                total = sum(c for _, _, c in current)
                if total < best_cost:
                    best_cost = total
                    best_assignment = list(current)
                return
            for node in nodes:
                cost = estimator.estimate(tasks[task_idx], node)
                current.append((tasks[task_idx], node, cost))
                try_assignment(task_idx + 1, current)
                current.pop()

        try_assignment(0, [])
        return best_assignment


if __name__ == "__main__":
    from node_descriptors import Node
    from task_descriptors import Task
    from src.true_estimates import TrueEstimator

    nodes = [Node(node_id="n1", capacity=10.0, location=(0.0, 0.0)),
             Node(node_id="n2", capacity=5.0, location=(1.0, 1.0))]
    tasks = [Task(task_id="t1", task_type="compute", size=4.0, deadline=100.0),
             Task(task_id="t2", task_type="compute", size=2.0, deadline=100.0)]

    est = TrueEstimator(seed=42)
    oracle = OracleBalancer(true_estimator=est)
    for task in tasks:
        n = oracle.assign(task, nodes)
        print(f"Oracle chose {n.node_id} for {task.task_id}")