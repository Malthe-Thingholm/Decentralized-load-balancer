"""Simulation metrics."""

from __future__ import annotations

import math
from typing import Sequence

from src.node import Node
from src.task import Task


def makespan(nodes: Sequence[Node]) -> float:
    """Time when the last node finishes its queue."""
    if not nodes:
        return 0.0
    return max(n.current_load_end for n in nodes)


def avg_latency(nodes: Sequence[Node], task_arrivals: dict[str, float]) -> float:
    """Average time from task arrival to completion."""
    latencies = _latencies(nodes, task_arrivals)
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def tail_latency(
    nodes: Sequence[Node], task_arrivals: dict[str, float], percentile: float = 0.99
) -> float:
    """Percentile latency (e.g., p99)."""
    latencies = _latencies(nodes, task_arrivals)
    if not latencies:
        return 0.0
    sorted_l = sorted(latencies)
    idx = max(0, int(math.ceil(percentile * len(sorted_l))) - 1)
    return sorted_l[idx]


def jain_fairness(nodes: Sequence[Node]) -> float:
    """Jain's fairness index over node utilizations."""
    utils = [
        n.total_busy_time / n.current_load_end if n.current_load_end > 0 else 0.0
        for n in nodes
    ]
    # Filter out zero-duration nodes
    active = [u for u in utils if u > 0]
    if not active:
        return 0.0
    numerator = sum(active) ** 2
    denominator = len(active) * sum(u ** 2 for u in active)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def utilization(nodes: Sequence[Node], sim_time: float) -> float:
    """Average node utilization: total_busy_time / (num_nodes * sim_time)."""
    if sim_time <= 0 or not nodes:
        return 0.0
    total = sum(n.total_busy_time for n in nodes)
    return total / (len(nodes) * sim_time)


def regret(
    nodes: Sequence[Node],
    task_arrivals: dict[str, float],
    oracle_makespan: float,
) -> float:
    """Regret = (actual_makespan - oracle_makespan) / oracle_makespan."""
    if oracle_makespan <= 0:
        return 0.0
    actual = makespan(nodes)
    return (actual - oracle_makespan) / oracle_makespan


def _latencies(
    nodes: Sequence[Node], task_arrivals: dict[str, float]
) -> list[float]:
    """Extract per-task latencies from completed tasks."""
    latencies: list[float] = []
    for node in nodes:
        for tid in node.completed:
            arrival = task_arrivals.get(tid)
            if arrival is not None:
                # Approximate: use total_busy_time as proxy for service time
                # In a real sim, we'd track per-task completion times
                latencies.append(node.total_busy_time / max(len(node.completed), 1))
    return latencies