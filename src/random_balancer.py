"""Random load balancer — picks a node uniformly at random."""

import random
from typing import Protocol

from src.node import Node
from src.task import Task


class RandomBalancer:
    """Assigns tasks to random nodes (no intelligence).
    Baseline for comparison."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        if not nodes:
            raise ValueError("No nodes available")
        return self._rng.choice(nodes)

    def on_complete(
        self, task: Task, node: Node, actual_cost: float
    ) -> None:
        pass


if __name__ == "__main__":
    from src.node import Node
    from src.task import Task

    nodes = [Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
             Node(id="n2", cpu_cap=5.0, memory_cap=5.0, network_cap=3.0)]
    task = Task(id="t1", cpu_req=1.0, memory_req=1.0, network_req=0.5)

    bal = RandomBalancer(seed=42)
    n = bal.assign(task, nodes)
    print(f"Random chose: {n.id}")