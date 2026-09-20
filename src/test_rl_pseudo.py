"""Pseudo-tests for RL Q-learning balancer.

Lightweight behavioral checks: cold-start, epsilon-greedy exploration,
Q-value updates, convergence toward best node, snapshot/merge.
"""

from __future__ import annotations

import sys

from src.rl_balancer import QLearningBalancer
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


def test_cold_start_all_nodes() -> None:
    """First assignments cover all nodes (epsilon-greedy with high epsilon)."""
    rl = QLearningBalancer(seed=42, epsilon=1.0, epsilon_decay=1.0)  # always explore
    nodes = make_nodes()
    task = make_task()
    choices = set()
    for _ in range(5):
        chosen = rl.assign(task, nodes)
        choices.add(chosen.id)
        rl.on_complete(task, chosen, 5.0)
    assert len(choices) >= 2, f"Expected at least 2 different nodes, got {choices}"
    print(f"PSEUDO-PASS: cold_start_all_nodes (choices: {choices})")


def test_epsilon_greedy() -> None:
    """With epsilon=0.0 and epsilon_min=0.0, always exploits (picks best Q-action)."""
    rl = QLearningBalancer(seed=42, epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0)
    nodes = make_nodes()
    task = make_task()

    # Force consistent state by setting queue depths to 0
    for n in nodes:
        rl._queue_depths[n.id] = 0
        rl._avg_costs[n.id] = 0.0

    state = rl._get_state(nodes)
    rl._set_q(state, 0, 10.0)  # n0: high Q
    rl._set_q(state, 1, 1.0)   # n1: low Q
    rl._set_q(state, 2, 1.0)   # n2: low Q

    # With epsilon=0 and epsilon_min=0, always picks the best Q-action
    for _ in range(10):
        chosen = rl.assign(task, nodes)
        assert chosen.id == "n0", f"Expected n0 (best Q), got {chosen.id}"
    print("PSEUDO-PASS: epsilon_greedy")


def test_q_value_update() -> None:
    """Q-values increase for low-cost actions, decrease for high-cost."""
    rl = QLearningBalancer(seed=42, epsilon=0.0, alpha=0.5)
    nodes = make_nodes()
    task = make_task()
    state = rl._get_state(nodes)

    # Initialize Q-values to 0
    rl._set_q(state, 0, 0.0)

    # Assign to n0 with low cost (reward = -3.0 / 20.0 = -0.15)
    chosen = nodes[0]
    rl.assign(task, nodes)
    rl.on_complete(task, chosen, actual_cost=3.0)
    q_after_low = rl._get_q(state, 0)

    # Reset and assign with high cost
    rl2 = QLearningBalancer(seed=42, epsilon=0.0, alpha=0.5)
    nodes2 = make_nodes()
    task2 = make_task()
    state2 = rl2._get_state(nodes2)
    rl2._set_q(state2, 0, 0.0)
    chosen2 = nodes2[0]
    rl2.assign(task2, nodes2)
    rl2.on_complete(task2, chosen2, actual_cost=15.0)
    q_after_high = rl2._get_q(state2, 0)

    # Low cost should give higher Q than high cost
    assert q_after_low > q_after_high, \
        f"Q after low cost ({q_after_low:.3f}) should exceed Q after high cost ({q_after_high:.3f})"
    print(f"PSEUDO-PASS: q_value_update (low_cost_q={q_after_low:.3f} > high_cost_q={q_after_high:.3f})")


def test_converges_to_best_node() -> None:
    """After enough training, prefers the node with lowest average cost."""
    from src.truesim import StochasticTrueModel

    model = StochasticTrueModel(seed=42, cost_scale=15.0)
    rl = QLearningBalancer(seed=42, epsilon=0.3, alpha=0.3, gamma=0.9,
                           epsilon_decay=0.995, epsilon_min=0.05)
    nodes = make_nodes()
    task = make_task()

    # Train for 100 steps
    for _ in range(100):
        chosen = rl.assign(task, nodes)
        cost = model.cost(task, chosen)
        rl.on_complete(task, chosen, cost)

    # Check which node was assigned most often (should be n0, the fastest)
    from collections import Counter
    counts = Counter(rl._assignments)
    assert counts["n0"] > counts["n1"], \
        f"n0 ({counts['n0']}) should be preferred over n1 ({counts['n1']})"
    print(f"PSEUDO-PASS: converges_to_best_node (counts: {dict(counts)})")


def test_snapshot_serializable() -> None:
    """Snapshot is JSON-serializable."""
    rl = QLearningBalancer(seed=42, num_nodes=3)
    nodes = make_nodes()
    task = make_task()
    rl.assign(task, nodes)
    rl.on_complete(task, rl.assign(task, nodes), 5.0)
    snap = rl.snapshot
    import json
    json.dumps(snap)
    assert "q_table_size" in snap
    assert "current_epsilon" in snap
    print("PSEUDO-PASS: snapshot_serializable")


def test_snapshot_subset() -> None:
    """Snapshot contains only essential fields (not full Q-table)."""
    rl = QLearningBalancer(seed=42, num_nodes=3)
    nodes = make_nodes()
    task = make_task()
    rl.assign(task, nodes)
    snap = rl.snapshot
    # Snapshot should NOT contain the full Q-table (could be huge)
    assert "q_table" not in snap, "Snapshot should not include full Q-table"
    assert "q_table_size" in snap
    print("PSEUDO-PASS: snapshot_subset")


def test_epsilon_decay() -> None:
    """Epsilon decays over time toward epsilon_min."""
    rl = QLearningBalancer(seed=42, epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1)
    nodes = make_nodes()
    task = make_task()

    initial_eps = rl._current_epsilon
    for _ in range(20):
        rl.assign(task, nodes)
        chosen = nodes[rl._last_action] if hasattr(rl, '_last_action') else nodes[0]
        rl.on_complete(task, chosen, 5.0)

    assert rl._current_epsilon < initial_eps, \
        f"Epsilon should decay: {initial_eps:.3f} -> {rl._current_epsilon:.3f}"
    assert rl._current_epsilon >= rl._epsilon_min, \
        f"Epsilon should not go below minimum: {rl._current_epsilon:.3f} < {rl._epsilon_min}"
    print(f"PSEUDO-PASS: epsilon_decay ({initial_eps:.3f} -> {rl._current_epsilon:.3f})")


def test_empty_nodes_error() -> None:
    """Assigning with empty node list raises ValueError."""
    rl = QLearningBalancer(seed=42)
    task = make_task()
    try:
        rl.assign(task, [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No nodes available" in str(e)
    print("PSEUDO-PASS: empty_nodes_error")


def main() -> int:
    print("=== RL Q-Learning Pseudo-Tests ===\n")
    tests = [
        test_cold_start_all_nodes,
        test_epsilon_greedy,
        test_q_value_update,
        test_converges_to_best_node,
        test_snapshot_serializable,
        test_snapshot_subset,
        test_epsilon_decay,
        test_empty_nodes_error,
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
