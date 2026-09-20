"""Oracle balancer — optimal online assignment using true costs.

The oracle makes the best possible decision for each arriving task given
current node states. It uses the TrueModel (ground-truth cost) to compute
the cost of assigning a task to each node, then picks the node that would
complete the task earliest: min over nodes of (next_available_time + cost).

This is the strongest fair online baseline — it sees true costs and current
queue state, but doesn't know future arrivals (same information constraint
as the online balancers, just with perfect cost knowledge).

For batch/offline optimal (global minimization), see batch_assign().

Type system: uses canonical src.node.Node and src.task.Task.
"""

from __future__ import annotations

from src.node import Node
from src.task import Task
from src.truesim import StochasticTrueModel


class OracleBalancer:
    """Optimal online assignment via true-cost-aware least-completion-time.

    For each task, picks the node that minimizes:
        node.next_available_time + true_model.cost(task, node)

    This is the best possible online decision with perfect cost knowledge.
    """

    def __init__(self, true_model: StochasticTrueModel | None = None):
        self._true_model = true_model

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Pick the node that minimizes the new makespan after assignment.

        For each candidate node, compute: if we assign this task here,
        what would the maximum completion time across ALL nodes be?
        Pick the node that results in the lowest maximum.

        This is a load-aware greedy strategy — it considers the global
        makespan impact, not just per-task completion time.
        """
        if not nodes:
            raise ValueError("No nodes available")

        best_node = None
        best_makespan = float("inf")

        for node in nodes:
            cost = self._true_model.cost(task, node) if self._true_model else 0.0
            # What would this node's completion time be?
            node_done = max(task.arrival_time, node.next_available_time) + cost
            # What would the system makespan be?
            other_max = max(
                (n.next_available_time for n in nodes if n.id != node.id),
                default=0.0,
            )
            system_makespan = max(node_done, other_max)
            if system_makespan < best_makespan:
                best_makespan = system_makespan
                best_node = node

        assert best_node is not None
        return best_node

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """Oracle doesn't need to learn — it always uses true costs."""
        _ = task, node, actual_cost

    def batch_assign(
        self,
        tasks: list[Task],
        nodes: list[Node],
        estimator: StochasticTrueModel,
    ) -> list[tuple[Task, Node, float]]:
        """Offline optimal: brute-force enumeration for small instances.

        Finds the assignment of tasks to nodes that minimizes total cost.
        Exponential in number of tasks — for reference only, not for live sim.

        Returns list of (task, node, cost) tuples.
        """
        n_tasks = len(tasks)
        n_nodes = len(nodes)

        if n_tasks == 0 or n_nodes == 0:
            return []

        if n_tasks > 10:
            import warnings

            warnings.warn(
                f"batch_assign with {n_tasks} tasks is exponential — "
                "slow and may hang. Use for small instances only.",
                stacklevel=2,
            )

        best_cost = float("inf")
        best_assignment: list[tuple[Task, Node, float]] = []

        def try_assignment(
            task_idx: int, current: list[tuple[Task, Node, float]]
        ) -> None:
            nonlocal best_cost, best_assignment
            if task_idx == n_tasks:
                total = sum(c for _, _, c in current)
                if total < best_cost:
                    best_cost = total
                    best_assignment = list(current)
                return
            for node in nodes:
                cost = estimator.cost(tasks[task_idx], node)
                current.append((tasks[task_idx], node, cost))
                try_assignment(task_idx + 1, current)
                current.pop()

        try_assignment(0, [])
        return best_assignment


if __name__ == "__main__":
    from src.node import Node
    from src.task import Task
    from src.truesim import StochasticTrueModel

    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=5.0, memory_cap=5.0, network_cap=3.0),
    ]
    tasks = [
        Task(id="t1", cpu_req=4.0, memory_req=4.0, network_req=2.0),
        Task(id="t2", cpu_req=2.0, memory_req=2.0, network_req=1.0),
    ]

    true_model = StochasticTrueModel(seed=42)
    oracle = OracleBalancer(true_model=true_model)

    for task in tasks:
        n = oracle.assign(task, nodes)
        print(f"Oracle chose {n.id} for {task.id}")
