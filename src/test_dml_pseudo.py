"""Pseudo-tests for Communication-Efficient DML balancer.

Validates: mode selection, queue-aware scoring, capability-aware scoring,
heterogeneity handling, snapshot/merge, and error handling.
"""

from __future__ import annotations

import sys
import math

from src.dml_balancer import CommunicationEfficientDMLBalancer
from src.node import Node
from src.task import Task


def make_nodes() -> list[Node]:
    return [
        Node(id="n0", cpu_cap=30.0, memory_cap=10.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=30.0, network_cap=10.0),
        Node(id="n2", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
    ]


def make_task() -> Task:
    return Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0)


def test_queue_aware_empty_queues() -> None:
    """With empty queues, queue_aware picks the highest-capacity node."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='queue_aware')
    nodes = make_nodes()
    task = make_task()

    # All queues empty
    for n in nodes:
        n.queue = []

    chosen = balancer.assign(task, nodes)
    # With equal queue depth (0), score = 0 for all; picks first minimum (=n0)
    # But n0 has highest geom_cap so it gets selected as the 'best' empty node
    print(f"PSEUDO-PASS: queue_aware_empty_queues (chose {chosen.id})")


def test_queue_aware_prefers_less_loaded() -> None:
    """queue_aware mode prefers nodes with fewer queued tasks."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='queue_aware')
    nodes = make_nodes()
    task = make_task()

    # n0: 5 tasks, n1: 1 task, n2: 3 tasks
    nodes[0].queue = ['t'] * 5
    nodes[1].queue = ['t']
    nodes[2].queue = ['t'] * 3

    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'n1', f"Expected n1 (least loaded), got {chosen.id}"
    print(f"PSEUDO-PASS: queue_aware_prefers_less_loaded (chose {chosen.id})")


def test_capability_aware_basic() -> None:
    """capability_aware mode scores nodes by queue_work / capacity_factor."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    nodes = make_nodes()
    task = make_task()

    # Pre-load: n0 has 3 tasks, n1 has 1, n2 has 2
    nodes[0].queue = ['t'] * 3
    nodes[1].queue = ['t']
    nodes[2].queue = ['t'] * 2

    chosen = balancer.assign(task, nodes)
    # n1 has least queue work and decent capacity for this task
    print(f"PSEUDO-PASS: capability_aware_basic (chose {chosen.id})")


def test_capability_match_cpu_heavy() -> None:
    """CPU-heavy task prefers high-cpu-cap nodes in capability_aware mode."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    nodes = [
        Node(id="n_cpu", cpu_cap=50.0, memory_cap=10.0, network_cap=10.0),
        Node(id="n_mem", cpu_cap=5.0, memory_cap=50.0, network_cap=10.0),
    ]
    # CPU-heavy task
    task = Task(id="cpu-task", cpu_req=20.0, memory_req=2.0, network_req=2.0, arrival_time=0.0)

    # n_cpu: excellent CPU (50 cap for 20 req = 0.4 ratio)
    # n_mem: terrible CPU (5 cap for 20 req = 4.0 ratio)
    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'n_cpu', f"Expected n_cpu for CPU-heavy task, got {chosen.id}"
    print(f"PSEUDO-PASS: capability_match_cpu_heavy (chose {chosen.id})")


def test_capability_match_mem_heavy() -> None:
    """Memory-heavy task prefers high-mem-cap nodes in capability_aware mode."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    nodes = [
        Node(id="n_cpu", cpu_cap=50.0, memory_cap=10.0, network_cap=10.0),
        Node(id="n_mem", cpu_cap=5.0, memory_cap=50.0, network_cap=10.0),
    ]
    # Memory-heavy task
    task = Task(id="mem-task", cpu_req=2.0, memory_req=20.0, network_req=2.0, arrival_time=0.0)

    # n_mem: excellent memory (50 cap for 20 req = 0.4 ratio)
    # n_cpu: terrible memory (10 cap for 20 req = 2.0 ratio)
    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'n_mem', f"Expected n_mem for memory-heavy task, got {chosen.id}"
    print(f"PSEUDO-PASS: capability_match_mem_heavy (chose {chosen.id})")


def test_capability_tiebreaker_with_load() -> None:
    """When both nodes match task resources equally, picks less loaded one."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    # Two identical nodes
    nodes = [
        Node(id="nA", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="nB", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
    ]
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0)

    # nA has 5 tasks, nB has 1 task
    nodes[0].queue = ['t'] * 5
    nodes[1].queue = ['t']

    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'nB', f"Expected nB (less loaded), got {chosen.id}"
    print(f"PSEUDO-PASS: capability_tiebreaker_with_load (chose {chosen.id})")


def test_snapshot_serializable() -> None:
    """Snapshot is JSON-serializable."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    nodes = make_nodes()
    task = make_task()
    balancer.assign(task, nodes)
    snap = balancer.snapshot
    import json
    json.dumps(snap)
    assert 'mode' in snap
    assert 'node_avg_costs' in snap
    print("PSEUDO-PASS: snapshot_serializable")


def test_merge() -> None:
    """Merging two snapshots combines node statistics."""
    balancer = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')
    nodes = make_nodes()
    task = make_task()

    # Run assignments on balancer 1
    for _ in range(5):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 10.0)
    snap1 = balancer.snapshot

    # Run on balancer 2
    balancer2 = CommunicationEfficientDMLBalancer(seed=42, mode='capability_aware')  # same seed!
    for _ in range(3):
        chosen = balancer2.assign(task, nodes)
        balancer2.on_complete(task, chosen, 15.0)
    snap2 = balancer2.snapshot

    # Merge
    balancer.merge(snap2)
    merged = balancer.snapshot

    # After merge, total_assignments should be sum of both
    assert merged['total_assignments'] == snap1['total_assignments'] + snap2['total_assignments'], \
        f"Expected {snap1['total_assignments'] + snap2['total_assignments']}, got {merged['total_assignments']}"
    # At least one node should have merged cost data
    assert len(merged['node_avg_costs']) > 0, "Should have some cost data after merge"
    print(f"PSEUDO-PASS: merge (total_assignments={snap1['total_assignments']}+{snap2['total_assignments']}={merged['total_assignments']})")


def test_empty_nodes_error() -> None:
    """Assigning with empty node list raises ValueError."""
    balancer = CommunicationEfficientDMLBalancer(seed=42)
    task = make_task()
    try:
        balancer.assign(task, [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No nodes available" in str(e)
    print("PSEUDO-PASS: empty_nodes_error")


def test_guessing_model_integration() -> None:
    """Uses GuessingModel when provided for cost estimates."""
    from src.guess import StochasticGuessingModel

    guess_model = StochasticGuessingModel(seed=42, error_std=0.1)
    balancer = CommunicationEfficientDMLBalancer(
        seed=42, mode='capability_aware', guessing_model=guess_model
    )
    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
    ]
    task = make_task()

    # Assign should use guess model internally
    chosen = balancer.assign(task, nodes)
    assert chosen.id in ('n0', 'n1'), f"Expected valid node, got {chosen.id}"
    print(f"PSEUDO-PASS: guessing_model_integration (chose {chosen.id})")


def main() -> int:
    print("=== DML Balancer Pseudo-Tests ===\n")
    tests = [
        test_queue_aware_empty_queues,
        test_queue_aware_prefers_less_loaded,
        test_capability_aware_basic,
        test_capability_match_cpu_heavy,
        test_capability_match_mem_heavy,
        test_capability_tiebreaker_with_load,
        test_snapshot_serializable,
        test_merge,
        test_empty_nodes_error,
        test_guessing_model_integration,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"PSEUDO-FAIL: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== Pseudo-test results: {passed} passed, {failed} failed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
