"""Power-of-two-choices load balancer — picks 2 random nodes, chooses the less loaded.

Classic result: this simple strategy dramatically reduces maximum load
compared to pure random, with O(log log n) max load scaling.
"""

import random
from typing import Protocol

from src.node import Node
from src.task import Task


class PowerOfTwoChoicesBalancer:
    """Power-of-two-choices: pick k=2 random nodes, assign to less-loaded.
    Load-aware, no central coordination needed.
    """

    def __init__(self, seed: int | None = None, k: int = 2):
        self._rng = random.Random(seed)
        self.k = k

    def _load_metric(self, node: Node) -> float:
        """How loaded this node is. Uses depth/capacity ratio."""
        if node.cpu_cap <= 0:
            return float('inf')
        return node.queue_depth / node.cpu_cap

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        if not nodes:
            raise ValueError("No nodes available")
        if len(nodes) <= self.k:
            # Fewer nodes than choices: pick the least loaded
            return min(nodes, key=self._load_metric)
        # Pick k random nodes, choose the least loaded among them
        candidates = random.sample(nodes, self.k)
        return min(candidates, key=self._load_metric)

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

    bal = PowerOfTwoChoicesBalancer(seed=42)
    n = bal.assign(task, nodes)
    print(f"Power-of-two chose: {n.id}")