"""Pseudo-tests for REPS sprayer.

Lightweight behavioral checks: uniform start, adaptation toward low-cost
nodes, floor enforcement, snapshot/merge, entropy tracking.
"""

from __future__ import annotations

import sys

from src.reps_balancer import REPSSprayer
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


def test_initial_uniform() -> None:
    """Initial distribution is uniform across all nodes."""
    sprayer = REPSSprayer(seed=42, num_nodes=3)
    nodes = make_nodes()
    task = make_task()

    # Access internal state via a few assignments
    for _ in range(3):
        sprayer.assign(task, nodes)

    probs = sprayer.snapshot["probs"]
    assert len(probs) == 3
    for p in probs.values():
        assert abs(p - 1.0 / 3) < 0.01, f"Expected ~0.333, got {p}"
    print("PSEUDO-PASS: initial_uniform")


def test_adapts_to_low_cost() -> None:
    """After observing low costs on n0 and high costs on n1, n0's
    probability increases."""
    from src.truesim import StochasticTrueModel

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    sprayer = REPSSprayer(seed=42, floor=0.05, adapt_rate=0.8, sensitivity=3.0)
    nodes = make_nodes()
    task = make_task()

    # Artificially give n0 low costs and n1 high costs
    for _ in range(10):
        # Force assignment to n0
        chosen = nodes[0]  # n0
        cost = model.cost(task, chosen)
        sprayer.assign(task, nodes)  # sample (might not be n0)
        sprayer.on_complete(task, chosen, cost)

    for _ in range(10):
        # Force assignment to n1
        chosen = nodes[1]  # n1
        cost = model.cost(task, chosen)
        sprayer.assign(task, nodes)
        sprayer.on_complete(task, chosen, cost)

    probs = sprayer.snapshot["probs"]
    # n0 should have higher probability than n1 (lower avg cost)
    assert probs["n0"] > probs["n1"], \
        f"n0 prob {probs['n0']:.3f} should exceed n1 prob {probs['n1']:.3f}"
    print(f"PSEUDO-PASS: adapts_to_low_cost (n0={probs['n0']:.3f} > n1={probs['n1']:.3f})")


def test_floor_enforced() -> None:
    """No node's probability drops below the floor (after normalization)."""
    sprayer = REPSSprayer(seed=42, num_nodes=3, floor=0.1, adapt_rate=1.0, sensitivity=5.0)
    nodes = make_nodes()
    task = make_task()

    # Run many tasks, all assigned to n0 (simulate forced assignment)
    for _ in range(20):
        chosen = nodes[0]
        sprayer.assign(task, nodes)
        sprayer.on_complete(task, chosen, 1.0)  # very low cost for n0

    probs = sprayer.snapshot["probs"]
    for nid, p in probs.items():
        assert p >= sprayer._floor - 0.001, \
            f"Node {nid} prob {p:.4f} below floor {sprayer._floor}"
    print(f"PSEUDO-PASS: floor_enforced (min prob={min(probs.values()):.3f}, floor={sprayer._floor})")


def test_entropy_tracking() -> None:
    """Entropy decreases as distribution becomes more peaked."""
    sprayer = REPSSprayer(seed=42, num_nodes=3, floor=0.01, adapt_rate=1.0, sensitivity=5.0)
    nodes = make_nodes()
    task = make_task()

    # Initialize distribution by assigning once (without on_complete, stays uniform)
    sprayer.assign(task, nodes)
    initial_entropy = sprayer.snapshot["entropy"]
    assert initial_entropy > 0.5, f"Initial entropy {initial_entropy:.2f} should be high (uniform, got {initial_entropy:.4f})"

    # Force all costs onto n0 → distribution peaks → entropy drops
    for _ in range(30):
        chosen = nodes[0]
        sprayer.assign(task, nodes)
        sprayer.on_complete(task, chosen, 1.0)

    final_entropy = sprayer.snapshot["entropy"]
    assert final_entropy < initial_entropy, \
        f"Entropy should decrease: {initial_entropy:.2f} -> {final_entropy:.2f}"
    print(f"PSEUDO-PASS: entropy_tracking ({initial_entropy:.2f} -> {final_entropy:.2f})")


def test_snapshot_serializable() -> None:
    """Snapshot is JSON-serializable."""
    sprayer = REPSSprayer(seed=42, num_nodes=3)
    nodes = make_nodes()
    task = make_task()
    sprayer.assign(task, nodes)
    snap = sprayer.snapshot
    import json
    json.dumps(snap)
    assert "probs" in snap
    assert "entropy" in snap
    print("PSEUDO-PASS: snapshot_serializable")


def test_merge() -> None:
    """Merging two sprayer snapshots combines statistics."""
    sprayer = REPSSprayer(seed=42, num_nodes=3, floor=0.05)
    nodes = make_nodes()
    task = make_task()

    # Run on sprayer 1
    for _ in range(5):
        chosen = sprayer.assign(task, nodes)
        sprayer.on_complete(task, chosen, 5.0)
    snap1 = sprayer.snapshot

    # Run on sprayer 2
    sprayer2 = REPSSprayer(seed=99, num_nodes=3, floor=0.05)
    for _ in range(3):
        chosen = sprayer2.assign(task, nodes)
        sprayer2.on_complete(task, chosen, 8.0)
    snap2 = sprayer2.snapshot

    # Merge
    sprayer.merge(snap2)
    merged = sprayer.snapshot

    assert merged["adaptations"] >= snap1["adaptations"]
    assert len(merged["avg_costs"]) >= len(snap1["avg_costs"])
    print(f"PSEUDO-PASS: merge (adaptations: {snap1['adaptations']} -> {merged['adaptations']})")


def test_empty_nodes_error() -> None:
    """Assigning with empty node list raises ValueError."""
    sprayer = REPSSprayer(seed=42)
    task = make_task()
    try:
        sprayer.assign(task, [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No nodes available" in str(e)
    print("PSEUDO-PASS: empty_nodes_error")


def main() -> int:
    print("=== REPS Pseudo-Tests ===\n")
    tests = [
        test_initial_uniform,
        test_adapts_to_low_cost,
        test_floor_enforced,
        test_entropy_tracking,
        test_snapshot_serializable,
        test_merge,
        test_empty_nodes_error,
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
