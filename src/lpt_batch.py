"""LPT (Longest Processing Time) batch baseline.

LPT is Graham's classic heuristic for makespan minimization on parallel
machines. It sorts all jobs by descending processing time, then assigns
each to the machine with the smallest current load.

For heterogeneous nodes, the generalization is:
1. Compute cost(task, node) for all task-node pairs
2. Sort tasks by their MINIMUM cost across all nodes (descending)
3. Assign each task to the node that gives the earliest completion time

This is an OFFLINE algorithm — it requires all tasks upfront. In our
simulation framework, it runs as a post-hoc comparison: given the same
task set that the online balancers saw, what makespan would LPT achieve?

Two cost variants:
- LPTTrue: uses TrueModel costs (the best LPT can do with perfect info)
- LPTGuess: uses GuessingModel costs (what LPT would see in practice)
"""

from __future__ import annotations

from typing import Protocol

from src.node import Node
from src.task import Task


class CostModel(Protocol):
    def cost(self, task: Task, node: Node) -> float: ...


class LPTBatch:
    """Offline LPT batch scheduler.

    Requires all tasks upfront. Sorts by descending min-cost across nodes,
    then greedily assigns each to the node with earliest completion.
    """

    def __init__(self, cost_model: CostModel):
        self._cost_model = cost_model

    def schedule(self, tasks: list[Task], nodes: list[Node]) -> dict[str, list[Task]]:
        """Run LPT on the full task set.

        Returns mapping: node_id -> list of assigned tasks (in assignment order).
        """
        if not nodes:
            raise ValueError("No nodes available")
        if not tasks:
            return {n.id: [] for n in nodes}

        # Compute min-cost for each task (across all nodes)
        task_min_cost: list[tuple[Task, float, list[tuple[Node, float]]]] = []
        for task in tasks:
            node_costs = [(n, self._cost_model.cost(task, n)) for n in nodes]
            min_cost = min(c for _, c in node_costs)
            task_min_cost.append((task, min_cost, node_costs))

        # Sort by descending min-cost (largest tasks first)
        task_min_cost.sort(key=lambda x: -x[1])

        # Track node completion times
        node_next_available: dict[str, float] = {n.id: 0.0 for n in nodes}
        assignment: dict[str, list[Task]] = {n.id: [] for n in nodes}

        for task, _min_cost, node_costs in task_min_cost:
            best_node = None
            best_completion = float("inf")

            for node, cost in node_costs:
                completion = max(task.arrival_time, node_next_available[node.id]) + cost
                if completion < best_completion:
                    best_completion = completion
                    best_node = node

            assert best_node is not None
            node_next_available[best_node.id] = best_completion
            assignment[best_node.id].append(task)

        return assignment

    def makespan(self, tasks: list[Task], nodes: list[Node]) -> float:
        """Compute the makespan LPT would achieve for this task set."""
        assignment = self.schedule(tasks, nodes)

        # Compute final completion time per node
        node_next_available: dict[str, float] = {n.id: 0.0 for n in nodes}

        # Re-run to get completion times (schedule() already computed them,
        # but we need them explicitly)
        task_min_cost: list[tuple[Task, float, list[tuple[Node, float]]]] = []
        for task in tasks:
            node_costs = [(n, self._cost_model.cost(task, n)) for n in nodes]
            min_cost = min(c for _, c in node_costs)
            task_min_cost.append((task, min_cost, node_costs))

        task_min_cost.sort(key=lambda x: -x[1])

        for task, _min_cost, node_costs in task_min_cost:
            best_node = None
            best_completion = float("inf")
            for node, cost in node_costs:
                completion = max(task.arrival_time, node_next_available[node.id]) + cost
                if completion < best_completion:
                    best_completion = completion
                    best_node = node
            assert best_node is not None
            node_next_available[best_node.id] = best_completion

        return max(node_next_available.values()) if node_next_available else 0.0


class LPTTrue(LPTBatch):
    """LPT with true costs — the strongest feasible offline baseline."""

    def __init__(self, true_model: StochasticTrueModel):
        super().__init__(cost_model=true_model)


class LPTGuess(LPTBatch):
    """LPT with guessed costs — what LPT would achieve with imperfect info."""

    def __init__(self, guess_model):
        super().__init__(cost_model=guess_model)


if __name__ == "__main__":
    from src.node import Node
    from src.task import Task
    from src.truesim import StochasticTrueModel
    from src.guess import StochasticGuessingModel  # noqa: F401 — for type hints only

    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=15.0, memory_cap=15.0, network_cap=8.0),
    ]
    tasks = [
        Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0),
        Task(id="t2", cpu_req=12.0, memory_req=8.0, network_req=4.0, arrival_time=0.0),
        Task(id="t3", cpu_req=3.0, memory_req=15.0, network_req=1.0, arrival_time=0.0),
        Task(id="t4", cpu_req=8.0, memory_req=4.0, network_req=6.0, arrival_time=0.0),
    ]

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    lpt = LPTTrue(model)

    for task in tasks:
        for n in nodes:
            print(f"  cost({task.id}, {n.id}) = {model.cost(task, n):.2f}")

    print()
    ms = lpt.makespan(tasks, nodes)
    print(f"LPT makespan: {ms:.2f}")

    assignment = lpt.schedule(tasks, nodes)
    for nid, tlist in assignment.items():
        print(f"  {nid}: {len(tlist)} tasks -> {[t.id for t in tlist]}")
