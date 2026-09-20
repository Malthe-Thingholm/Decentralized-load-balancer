"""Lower bound calculator for optimal makespan.

These are provable lower bounds — no algorithm can beat them. They form
the "oracle" baseline without requiring expensive offline optimization.

Two lower bounds:
1. Work bound: total CPU-seconds / num_nodes (ignores scheduling overhead)
2. Max-task bound: largest single task cost (you can't start it faster)
3. Combined: max(work_bound, max_task_bound) — standard LPT lower bound

The combined bound is O(n·m) to compute and is used in scheduling theory
as a proxy for the optimal makespan when exact optimization is infeasible
(see Graham 1969, LPT survey literature).
"""

from __future__ import annotations

from typing import Protocol

from src.node import Node
from src.task import Task


class CostModel(Protocol):
    def cost(self, task: Task, node: Node) -> float: ...


def work_bound(tasks: list[Task], nodes: list[Node], cost_model: CostModel) -> float:
    """Total work divided by number of nodes.

    Idea: even if we could split every task perfectly across all nodes,
    you still need (sum of all work) / (number of nodes) time units.
    This ignores heterogeneity, task indivisibility, and scheduling delays.
    """
    if not nodes or not tasks:
        return 0.0

    total_work = 0.0
    for task in tasks:
        # Best-case: assign each task to its cheapest node
        best_cost = min(cost_model.cost(task, n) for n in nodes)
        total_work += best_cost

    return total_work / len(nodes)


def max_task_bound(tasks: list[Task], nodes: list[Node], cost_model: CostModel) -> float:
    """Largest single task's minimum cost across all nodes.

    Idea: no schedule can finish before the largest task is done, and the
    smallest it can possibly take is its cheapest-node cost.
    """
    if not nodes or not tasks:
        return 0.0

    max_min_cost = 0.0
    for task in tasks:
        best_cost = min(cost_model.cost(task, n) for n in nodes)
        max_min_cost = max(max_min_cost, best_cost)

    return max_min_cost


def combined_bound(tasks: list[Task], nodes: list[Node], cost_model: CostModel) -> float:
    """max(work_bound, max_task_bound) — standard LPT lower bound.

    This is the tightest tractable lower bound used in scheduling theory
    (Graham 1969, "Bounds on the relative efficiency...").
    It is a valid lower bound on OPT for the offline makespan minimization
    problem, even on heterogeneous machines.
    """
    return max(work_bound(tasks, nodes, cost_model), max_task_bound(tasks, nodes, cost_model))


def relative_gap(makespan: float, lower_bound: float) -> float:
    """(makespan - lb) / lb — always >= 0 when lb is valid."""
    if lower_bound <= 0:
        return float("inf")
    return (makespan - lower_bound) / lower_bound


if __name__ == "__main__":
    from src.truesim import StochasticTrueModel

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

    wb = work_bound(tasks, nodes, model)
    mtb = max_task_bound(tasks, nodes, model)
    cb = combined_bound(tasks, nodes, model)

    print(f"Work bound:       {wb:.2f}")
    print(f"Max-task bound:   {mtb:.2f}")
    print(f"Combined (LPT LB): {cb:.2f}")

    from src.lpt_batch import LPTTrue
    lpt = LPTTrue(model)
    ms = lpt.makespan(tasks, nodes)
    print(f"\nLPT makespan:      {ms:.2f}")
    print(f"LPT vs lower bound: gap = {relative_gap(ms, cb):.3f} ({(1 + relative_gap(ms, cb)) * 100:.1f}% of LB)")
