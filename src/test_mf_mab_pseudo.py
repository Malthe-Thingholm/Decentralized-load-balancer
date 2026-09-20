"""Pseudo-tests for Mean-Field MAB balancer.

Lightweight behavioral checks that don't require a full simulation run.
These validate the bandit logic: cold-start, exploration, exploitation,
snapshot/merge, and error handling.
"""

from __future__ import annotations

import sys

from src.mf_mab_balancer import MeanFieldMABBalancer
from src.node import Node
from src.task import Task


def make_nodes() -> list[Node]:
    return [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=15.0, memory_cap=15.0, network_cap=8.0),
    ]


def make_task() -> Task:
    return Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0, arrival_time=0.0)


def test_cold_start() -> None:
    """First assignment goes to first node (all arms have 0 pulls)."""
    balancer = MeanFieldMABBalancer(seed=42)
    nodes = make_nodes()
    task = make_task()
    chosen = balancer.assign(task, nodes)
    assert chosen.id == "n0", f"Expected n0, got {chosen.id}"
    print("PSEUDO-PASS: cold_start")


def test_balances_initially() -> None:
    """First 3 assignments cover all nodes (cold start round-robin)."""
    balancer = MeanFieldMABBalancer(seed=42)
    nodes = make_nodes()
    task = make_task()
    choices = []
    for _ in range(3):
        chosen = balancer.assign(task, nodes)
        choices.append(chosen.id)
        # Simulate completion with fake cost
        balancer.on_complete(task, chosen, 5.0)
    assert set(choices) == {"n0", "n1", "n2"}, f"Expected all 3 nodes, got {choices}"
    print("PSEUDO-PASS: balances_initially")


def test_exploits_best_node() -> None:
    """After enough samples, prefers the best (lowest-cost) node."""
    from src.truesim import StochasticTrueModel

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    balancer = MeanFieldMABBalancer(seed=42, normalize_costs=True)
    nodes = make_nodes()
    task = make_task()

    # Run 50 tasks to let the bandit converge
    for _ in range(50):
        chosen = balancer.assign(task, nodes)
        cost = model.cost(task, chosen)
        balancer.on_complete(task, chosen, cost)

    # n0 is fastest → should get most assignments
    counts = {nid: balancer.snapshot["arm_stats"][nid]["count"]
              for nid in balancer.snapshot["arm_stats"]}
    assert counts["n0"] > counts["n1"], \
        f"n0 should be preferred over n1: {counts}"
    assert counts["n0"] > counts["n2"], \
        f"n0 should be preferred over n2: {counts}"
    print(f"PSEUDO-PASS: exploits_best_node (counts: {counts})")


def test_snapshot_serializable() -> None:
    """Snapshot produces a JSON-serializable dict."""
    balancer = MeanFieldMABBalancer(seed=42)
    nodes = make_nodes()
    task = make_task()
    balancer.assign(task, nodes)
    snap = balancer.snapshot
    assert isinstance(snap, dict)
    assert "total_pulls" in snap
    assert "arm_stats" in snap
    # Verify all values are JSON-serializable
    import json
    json.dumps(snap)  # would raise if not serializable
    print(f"PSEUDO-PASS: snapshot_serializable")


def test_merge() -> None:
    """Merging two snapshots combines statistics."""
    balancer = MeanFieldMABBalancer(seed=42)
    nodes = make_nodes()
    task = make_task()

    # Run a few assignments on balancer 1
    for _ in range(3):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 5.0)

    snap1 = balancer.snapshot

    # Create a second balancer with different stats
    balancer2 = MeanFieldMABBalancer(seed=99)
    for _ in range(2):
        chosen = balancer2.assign(task, nodes)
        balancer2.on_complete(task, chosen, 8.0)

    snap2 = balancer2.snapshot

    # Merge snap2 into balancer1
    balancer.merge(snap2)

    merged = balancer.snapshot
    assert merged["total_pulls"] >= snap1["total_pulls"]
    assert "arm_stats" in merged
    print(f"PSEUDO-PASS: merge (total_pulls: {snap1['total_pulls']} + {snap2['total_pulls']} -> {merged['total_pulls']})")


def test_empty_nodes_error() -> None:
    """Assigning with empty node list raises ValueError."""
    balancer = MeanFieldMABBalancer(seed=42)
    task = make_task()
    try:
        balancer.assign(task, [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No nodes available" in str(e)
    print("PSEUDO-PASS: empty_nodes_error")


def test_normalize_costs() -> None:
    """Cost normalization maps costs to rewards in [0, 1]."""
    balancer = MeanFieldMABBalancer(seed=42, normalize_costs=True,
                                     min_cost=0.0, max_cost=100.0)
    nodes = make_nodes()
    task = make_task()

    # Assign to n0 with cost 10
    chosen = balancer.assign(task, nodes)
    balancer.on_complete(task, chosen, actual_cost=10.0)
    assert balancer._arm_stats["n0"]["count"] == 1

    # Assign to n1 with cost 90
    chosen = balancer.assign(task, nodes)
    balancer.on_complete(task, chosen, actual_cost=90.0)
    assert balancer._arm_stats["n1"]["count"] == 1

    # n0 (cost 10) should have higher reward than n1 (cost 90)
    r0 = balancer._arm_stats["n0"]["sum_reward"]
    r1 = balancer._arm_stats["n1"]["sum_reward"]
    assert r0 > r1, f"n0 reward {r0} should exceed n1 reward {r1}"
    print(f"PSEUDO-PASS: normalize_costs (r0={r0:.3f} > r1={r1:.3f})")


def main() -> int:
    print("=== MF-MAB Pseudo-Tests ===\n")
    tests = [
        test_cold_start,
        test_balances_initially,
        test_exploits_best_node,
        test_snapshot_serializable,
        test_merge,
        test_empty_nodes_error,
        test_normalize_costs,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"PSEUDO-FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n=== Pseudo-test results: {passed} passed, {failed} failed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
