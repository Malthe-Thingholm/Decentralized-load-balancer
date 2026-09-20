"""List Scheduling — online greedy baseline.

Assigns each arriving task to the node that would complete it earliest
(next_available_time + cost). This is Graham's classic list scheduling
algorithm, extended to heterogeneous nodes.

Two cost variants:
- True costs: uses the TrueModel (oracle visibility, but online decisions)
- Guessed costs: uses GuessingModel (same info as the real balancers)

For the offline sorted variant (LPT), see src.lpt_batch.
"""

from __future__ import annotations

from src.node import Node
from src.task import Task
from src.truesim import StochasticTrueModel
from src.guess import StochasticGuessingModel


class ListScheduling:
    """Online list scheduling: assign each task to earliest-completing node.

    Graham's classic heuristic for makespan minimization. For heterogeneous
    nodes, the generalization is: pick the node that minimizes
        node.next_available_time + cost(task, node)
    where cost comes from either the TrueModel or a GuessingModel.

    This is an ONLINE algorithm — it sees one task at a time and makes
    irrevocable decisions, same as the real balancers.
    """

    def __init__(
        self,
        cost_model,
        seed: int = 42,
    ):
        """Parameters
        ----------
        cost_model: StochasticTrueModel | StochasticGuessingModel | None
            If None, falls back to queue-depth-only (least connections).
        seed: RNG seed (unused — list scheduling is deterministic given costs).
        """
        self._cost_model = cost_model
        self._seed = seed

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        """Assign *task* to the node with earliest expected completion."""
        if not nodes:
            raise ValueError("No nodes available")

        if self._cost_model is None:
            # Fallback: least queue depth (least connections)
            return min(nodes, key=lambda n: n.queue_depth)

        best_node = None
        best_completion = float("inf")

        for node in nodes:
            cost = self._cost_model.cost(task, node)
            completion = max(task.arrival_time, node.next_available_time) + cost
            if completion < best_completion:
                best_completion = completion
                best_node = node

        assert best_node is not None
        return best_node

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        """List scheduling doesn't learn — it uses the cost model directly."""
        _ = task, node, actual_cost


class ListSchedulingTrue(ListScheduling):
    """List scheduling with true costs (oracle visibility, online decisions)."""

    def __init__(self, true_model: StochasticTrueModel, seed: int = 42):
        super().__init__(cost_model=true_model, seed=seed)


class ListSchedulingGuess(ListScheduling):
    """List scheduling with guessed costs (same info as real balancers)."""

    def __init__(self, guess_model: StochasticGuessingModel, seed: int = 42):
        super().__init__(cost_model=guess_model, seed=seed)


if __name__ == "__main__":
    from src.node import Node
    from src.task import Task
    from src.truesim import StochasticTrueModel

    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
    ]
    tasks = [
        Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0),
        Task(id="t2", cpu_req=8.0, memory_req=8.0, network_req=3.0, arrival_time=1.0),
    ]

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    ls = ListSchedulingTrue(model)

    for task in tasks:
        chosen = ls.assign(task, nodes)
        cost = model.cost(task, chosen)
        start = max(task.arrival_time, chosen.next_available_time)
        done = start + cost
        chosen.next_available_time = done
        chosen.assign(task.id)
        print(f"  {task.id} -> {chosen.id}, cost={cost:.2f}, done={done:.2f}")
