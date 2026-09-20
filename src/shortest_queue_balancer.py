"""Shortest-queue load balancer — assigns to node with fewest pending tasks.

Simple load-aware baseline: uses queue depth only (no capacity awareness).
"""

from src.node import Node
from src.task import Task


class ShortestQueueBalancer:
    """Assigns tasks to node with shallowest queue.
    Load-aware but ignores capacity differences."""

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        if not nodes:
            raise ValueError("No nodes available")
        return min(nodes, key=lambda n: n.queue_depth)

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

    bal = ShortestQueueBalancer()
    n = bal.assign(task, nodes)
    print(f"Shortest-queue chose: {n.id}")