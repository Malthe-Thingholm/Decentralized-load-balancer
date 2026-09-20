"""Round-robin load balancer — cycles through nodes in order.

Simple deterministic baseline: ignores current load state.
"""

from src.node import Node
from src.task import Task


class RoundRobinBalancer:
    """Assigns tasks to nodes in round-robin order.
    No load-awareness."""

    def __init__(self) -> None:
        self._idx = 0

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        if not nodes:
            raise ValueError("No nodes available")
        node = nodes[self._idx % len(nodes)]
        self._idx += 1
        return node

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

    bal = RoundRobinBalancer()
    n = bal.assign(task, nodes)
    print(f"Round-robin chose: {n.id}")