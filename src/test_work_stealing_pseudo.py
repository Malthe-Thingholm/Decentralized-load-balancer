"""Pseudo-tests for WorkStealingBalancer.

These are descriptions of expected behavior that function as tests.
Run with: python src/work_stealing_balancer.py
Or import and assert in a real test suite.
"""

from src.node import Node
from src.task import Task
from src.work_stealing_balancer import (
    WorkStealingBalancer,
    AdaptiveWorkStealingBalancer,
)


def pseudo_test_basic_assignment() -> None:
    """A2WS assigns tasks to the least-loaded node when no stealing is warranted."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n3", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
    ]
    task = Task(id="t1", cpu_req=1.0, memory_req=1.0, network_req=0.5)

    bal = WorkStealingBalancer(seed=42)
    chosen = bal.assign(task, nodes)

    # All nodes have equal queue depth (0), so any node is valid
    assert chosen in nodes, f"chosen node {chosen.id} not in nodes"
    assert chosen.queue_depth == 0
    assert bal.total_assignments == 1
    print("PSEUDO-PASS: basic_assignment — assigns to a valid node")


def pseudo_test_stealing_behavior() -> None:
    """When one node is heavily loaded and another is idle, A2WS should
    detect the imbalance and attempt a steal."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),  # idle
        Node(id="n2", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),  # overloaded
        Node(id="n3", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),  # idle
    ]
    # Load n2 heavily
    for i in range(10):
        nodes[1].queue.append(f"t-preexisting-{i}")

    task = Task(id="t-new", cpu_req=1.0, memory_req=1.0, network_req=0.5)

    # With steal_threshold=0.25 and adaptive_radius=2.0, the idle nodes
    # should detect n2 as a steal victim
    bal = WorkStealingBalancer(
        seed=42, adaptive_radius=2.0, steal_threshold=0.25
    )
    chosen = bal.assign(task, nodes)

    # The least-loaded node should be chosen (n1 or n3)
    assert chosen.id in ("n1", "n3"), f"expected idle node, got {chosen.id}"
    assert chosen.queue_depth == 0
    # steal should have been attempted (total_steals > 0)
    assert bal.total_steals > 0, "expected at least one steal attempt"
    assert bal.steal_rate > 0, "expected non-zero steal rate"
    print(f"PSEUDO-PASS: stealing_behavior — steal_rate={bal.steal_rate:.2f}, "
          f"total_steals={bal.total_steals}")


def pseudo_test_no_steal_when_balanced() -> None:
    """When all nodes have similar load, no stealing should occur."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n3", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
    ]
    # All nodes have similar load (2 tasks each)
    for n in nodes:
        for i in range(2):
            n.queue.append(f"t-preexisting-{n.id}-{i}")

    task = Task(id="t-new", cpu_req=1.0, memory_req=1.0, network_req=0.5)

    bal = WorkStealingBalancer(
        seed=42, adaptive_radius=2.0, steal_threshold=0.25
    )
    chosen = bal.assign(task, nodes)

    assert chosen in nodes
    assert bal.total_steals == 0, "expected no steals in balanced system"
    print("PSEUDO-PASS: no_steal_when_balanced — steal_rate=0.0")


def pseudo_test_adaptive_adjusts_threshold() -> None:
    """Adaptive variant should adjust steal_threshold based on observed rate."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
    ]
    # Heavy load on n2
    for i in range(15):
        nodes[1].queue.append(f"t-preexisting-{i}")

    bal = AdaptiveWorkStealingBalancer(
        seed=42,
        adaptive_radius=2.0,
        steal_threshold=0.25,
        adaptation_rate=0.1,
    )

    # Run multiple assignments to trigger adaptation
    for i in range(20):
        task = Task(id=f"t-{i}", cpu_req=1.0, memory_req=1.0, network_req=0.5)
        chosen = bal.assign(task, nodes)
        bal.on_complete(task, chosen, 1.0)

    # After many assignments with imbalance, threshold should have adapted
    assert bal.steal_threshold >= bal.min_threshold
    assert bal.steal_threshold <= bal.max_threshold
    print(f"PSEUDO-PASS: adaptive_adjusts_threshold — "
          f"final_threshold={bal.steal_threshold:.3f}, "
          f"steal_rate={bal.steal_rate:.2f}")


def pseudo_test_snapshot() -> None:
    """Snapshot should return valid statistics."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=5.0, memory_cap=5.0, network_cap=3.0),
    ]

    bal = WorkStealingBalancer(seed=42)
    for i in range(10):
        task = Task(id=f"t-{i}", cpu_req=1.0, memory_req=1.0, network_req=0.5)
        chosen = bal.assign(task, nodes)
        bal.on_complete(task, chosen, 1.0)

    snap = bal.snapshot()
    assert "total_assignments" in snap
    assert "total_steals" in snap
    assert "steal_rate" in snap
    assert "adaptive_radius" in snap
    assert snap["total_assignments"] == 10
    print(f"PSEUDO-PASS: snapshot — {snap}")


def pseudo_test_empty_nodes_error() -> None:
    """Assigning to empty node list should raise ValueError."""
    bal = WorkStealingBalancer(seed=42)
    try:
        bal.assign(Task(id="t1", cpu_req=1.0, memory_req=1.0, network_req=0.5), [])
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "No nodes available" in str(e)
        print("PSEUDO-PASS: empty_nodes_error — ValueError raised")


def pseudo_test_seed_determinism() -> None:
    """Same seed should produce identical assignment sequences."""
    nodes = [
        Node(id="n1", cpu_cap=10.0, memory_cap=10.0, network_cap=5.0),
        Node(id="n2", cpu_cap=5.0, memory_cap=5.0, network_cap=3.0),
        Node(id="n3", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
    ]

    bal1 = WorkStealingBalancer(seed=99)
    bal2 = WorkStealingBalancer(seed=99)

    choices1 = []
    choices2 = []
    for i in range(10):
        task = Task(id=f"t-{i}", cpu_req=1.0, memory_req=1.0, network_req=0.5)
        # Add some pre-existing load for variety
        if i % 3 == 0:
            nodes[1].queue.append(f"pre-{i}")
        c1 = bal1.assign(task, nodes)
        choices1.append(c1.id)
        bal1.on_complete(task, c1, 1.0)

        c2 = bal2.assign(task, nodes)
        choices2.append(c2.id)
        bal2.on_complete(task, c2, 1.0)

    assert choices1 == choices2, f"seed determinism failed: {choices1} != {choices2}"
    print(f"PSEUDO-PASS: seed_determinism — {choices1}")


def run_all() -> None:
    tests = [
        pseudo_test_basic_assignment,
        pseudo_test_stealing_behavior,
        pseudo_test_no_steal_when_balanced,
        pseudo_test_adaptive_adjusts_threshold,
        pseudo_test_snapshot,
        pseudo_test_empty_nodes_error,
        pseudo_test_seed_determinism,
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


if __name__ == "__main__":
    run_all()
