"""Balancer interface."""

from __future__ import annotations

from typing import Protocol

from src.node import Node
from src.task import Task


class Balancer(Protocol):
    """Assigns tasks to nodes; notified on completions."""

    def assign(self, task: Task, nodes: list[Node]) -> Node:
        ...

    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
        ...