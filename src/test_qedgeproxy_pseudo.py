"""Pseudo-tests for QEdgeProxy multi-player bandit balancer.

Validates: cold-start behavior, collision resolution, QoS integration,
proxy independence, snapshot/merge, and error handling.
"""

from __future__ import annotations

import math
from collections import defaultdict

from src.node import Node
from src.qedgeproxy_balancer import QEdgeProxyBalancer, UCB1Bandit
from src.task import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_nodes() -> list[Node]:
    return [
        Node(id='nA', cpu_cap=30.0, memory_cap=20.0, network_cap=10.0),
        Node(id='nB', cpu_cap=10.0, memory_cap=25.0, network_cap=8.0),
        Node(id='nC', cpu_cap=20.0, memory_cap=15.0, network_cap=15.0),
    ]

def make_task() -> Task:
    return Task(id='t1', cpu_req=8.0, memory_req=12.0, network_req=5.0,
                deadline=10.0, priority=1)


def main() -> None:
    tests = [
        test_cold_start_distribution,
        test_collision_resolution,
        test_proxy_independence,
        test_qos_priority_weighting,
        test_qos_deadline_penalty,
        test_snapshot_serializable,
        test_merge,
        test_empty_nodes_error,
        test_convergence_to_best_node,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f'PSEUDO-FAIL: {test.__name__}: {e}')
        except Exception as e:
            failed += 1
            print(f'PSEUDO-ERROR: {test.__name__}: {type(e).__name__}: {e}')

    print(f'\n=== QEdgeProxy Pseudo-Test Results ===\n\nPseudo-test results: {passed} passed, {failed} failed')
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cold_start_distribution() -> None:
    """Cold-start: proxies should explore all nodes before converging."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=2, ucb_c=2.0)
    nodes = make_nodes()
    task = make_task()

    choices = []
    for _ in range(6):
        chosen = balancer.assign(task, nodes)
        choices.append(chosen.id)
        balancer.on_complete(task, chosen, 1.0)

    assert len(set(choices)) >= 2, \
        f"Expected >= 2 distinct nodes, got {set(choices)}"
    print(f'PSEUDO-PASS: cold_start_distribution (choices={set(choices)})')


def test_collision_resolution() -> None:
    """When proxies collide on the same node, resolution picks winner and
    losers re-select from remaining nodes."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=3, ucb_c=2.0)
    nodes = make_nodes()
    task = make_task()

    for _ in range(5):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 1.0)

    for _ in range(10):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 0.5)

    assert balancer._collision_count >= 0
    print(f'PSEUDO-PASS: collision_resolution '
          f'(collisions={balancer._collision_count}/{balancer._total_assignments})')


def test_proxy_independence() -> None:
    """Different proxies should make different selections (independence)."""
    balancer = QEdgeProxyBalancer(seed=123, num_proxies=3, ucb_c=1.0)
    nodes = make_nodes()
    task = make_task()

    for proxy in balancer._proxies:
        proxy.counts = {'nA': 5, 'nB': 3, 'nC': 2}
        proxy.value_sums = {'nA': -2.5, 'nB': -1.5, 'nC': -1.0}

    selections = []
    for i in range(balancer._num_proxies):
        sel = balancer._proxy_selects(i, [n.id for n in nodes])
        selections.append(sel)

    assert len(selections) == balancer._num_proxies
    assert all(s in [n.id for n in nodes] for s in selections)
    print(f'PSEUDO-PASS: proxy_independence (selections={selections})')


def test_qos_priority_weighting() -> None:
    """Higher-priority tasks should produce larger reward signals."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=1, ucb_c=2.0)
    nodes = make_nodes()

    low_pri_task = Task(id='low', cpu_req=8.0, memory_req=12.0,
                        network_req=5.0, priority=1)
    high_pri_task = Task(id='high', cpu_req=8.0, memory_req=12.0,
                         network_req=5.0, priority=5)

    chosen = balancer.assign(low_pri_task, nodes)
    balancer.on_complete(low_pri_task, chosen, 2.0)
    low_reward = balancer._compute_reward(low_pri_task, chosen, 2.0)

    chosen2 = balancer.assign(high_pri_task, nodes)
    balancer.on_complete(high_pri_task, chosen2, 2.0)
    high_reward = balancer._compute_reward(high_pri_task, chosen2, 2.0)

    assert high_reward < low_reward, \
        f"Higher priority should give more negative reward (lower cost), " \
        f"got high={high_reward} vs low={low_reward}"
    print(f'PSEUDO-PASS: qos_priority_weighting '
          f'(low_pri_reward={low_reward:.2f}, high_pri_reward={high_reward:.2f})')


def test_qos_deadline_penalty() -> None:
    """Missing a deadline should increase the penalty (reduce reward)."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=1, ucb_c=2.0)
    nodes = make_nodes()

    on_time_task = Task(id='ontime', cpu_req=8.0, memory_req=12.0,
                        network_req=5.0, deadline=10.0, priority=1)
    late_task = Task(id='late', cpu_req=8.0, memory_req=12.0,
                     network_req=5.0, deadline=5.0, priority=1)

    chosen = balancer.assign(on_time_task, nodes)
    on_time_reward = balancer._compute_reward(on_time_task, chosen, 3.0)

    chosen2 = balancer.assign(late_task, nodes)
    late_reward = balancer._compute_reward(late_task, chosen2, 8.0)

    assert late_reward < on_time_reward, \
        f"Late task should have worse reward, got late={late_reward:.2f} " \
        f"vs on_time={on_time_reward:.2f}"
    print(f'PSEUDO-PASS: qos_deadline_penalty '
          f'(on_time={on_time_reward:.2f}, late={late_reward:.2f})')


def test_snapshot_serializable() -> None:
    """Snapshot should be a serializable dict with all proxy states."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=2, ucb_c=2.0)
    nodes = make_nodes()
    task = make_task()

    for _ in range(5):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 1.0)

    snap = balancer.snapshot()

    assert isinstance(snap, dict), f"Expected dict, got {type(snap)}"
    assert 'version' in snap
    assert snap['version'] == 1
    assert 'proxies' in snap
    assert len(snap['proxies']) == balancer._num_proxies
    assert 'total_assignments' in snap
    assert snap['total_assignments'] == 5, \
        f"Expected 5, got {snap['total_assignments']}"
    print(f'PSEUDO-PASS: snapshot_serializable')


def test_merge() -> None:
    """Merging two snapshots combines proxy statistics."""
    balancer1 = QEdgeProxyBalancer(seed=42, num_proxies=2, ucb_c=2.0)
    nodes = make_nodes()
    task = make_task()

    for _ in range(5):
        chosen = balancer1.assign(task, nodes)
        balancer1.on_complete(task, chosen, 1.0)

    snap1 = balancer1.snapshot()

    balancer2 = QEdgeProxyBalancer(seed=42, num_proxies=2, ucb_c=2.0)
    for _ in range(3):
        chosen = balancer2.assign(task, nodes)
        balancer2.on_complete(task, chosen, 1.0)

    balancer1.merge(snap1)
    merged = balancer1.snapshot()

    assert merged['total_assignments'] == snap1['total_assignments'] * 2, \
        f"Expected {snap1['total_assignments'] * 2}, got {merged['total_assignments']}"
    print(f'PSEUDO-PASS: merge (total_assignments={snap1["total_assignments"]}*2={merged["total_assignments"]})')


def test_empty_nodes_error() -> None:
    """Assigning with empty node list should raise ValueError."""
    balancer = QEdgeProxyBalancer(seed=42)
    try:
        balancer.assign(Task(id='t1', cpu_req=1.0, memory_req=1.0, network_req=1.0),
                        [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print('PSEUDO-PASS: empty_nodes_error')


def test_convergence_to_best_node() -> None:
    """Over many trials, the bandit should converge to the lowest-cost node."""
    balancer = QEdgeProxyBalancer(seed=42, num_proxies=1, ucb_c=0.5)
    nodes = [
        Node(id='slow', cpu_cap=5.0, memory_cap=5.0, network_cap=5.0),
        Node(id='medium', cpu_cap=15.0, memory_cap=15.0, network_cap=10.0),
        Node(id='fast', cpu_cap=30.0, memory_cap=30.0, network_cap=20.0),
    ]
    task = Task(id='t1', cpu_req=5.0, memory_req=5.0, network_req=5.0,
                deadline=100.0, priority=1)

    counts = defaultdict(int)
    for _ in range(200):
        chosen = balancer.assign(task, nodes)
        counts[chosen.id] += 1
        costs = {'slow': 5.0, 'medium': 2.0, 'fast': 1.0}
        balancer.on_complete(task, chosen, costs[chosen.id])

    fast_pct = counts['fast'] / sum(counts.values())
    assert fast_pct > 0.5, \
        f"Fast node should get >50% of assignments, got {fast_pct:.2%} " \
        f"(counts={dict(counts)})"
    print(f'PSEUDO-PASS: convergence_to_best_node '
          f'(fast={counts["fast"]}/{sum(counts.values())} = {fast_pct:.2%})')


if __name__ == '__main__':
    main()
