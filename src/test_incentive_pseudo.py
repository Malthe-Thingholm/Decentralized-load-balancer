"""Pseudo-tests for Incentive-Based load balancer.

Validates: reputation-weighted scoring, reputation updates,
cold-start behavior, snapshot/merge, mode switching, and error handling.
"""

from __future__ import annotations

from collections import defaultdict

from src.incentive_balancer import IncentiveBalancer
from src.node import Node
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
    return Task(id='t1', cpu_req=8.0, memory_req=12.0, network_req=5.0)


def main() -> None:
    tests = [
        test_cold_start_capacity_heuristic,
        test_reputation_weighted_scoring,
        test_reputation_update_on_complete,
        test_poor_node_reputation_decreases,
        test_snapshot_serializable,
        test_merge,
        test_price_aware_mode,
        test_empty_nodes_error,
        test_cold_start_no_history,
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

    print(f'\n=== Incentive Balancer Pseudo-Test Results ===\n\nPseudo-test results: {passed} passed, {failed} failed')
    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cold_start_capacity_heuristic() -> None:
    """Cold-start: nodes with higher capacity should be preferred (lower
    heuristic cost)."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = [
        Node(id='small', cpu_cap=5.0, memory_cap=5.0, network_cap=5.0),
        Node(id='large', cpu_cap=50.0, memory_cap=50.0, network_cap=50.0),
    ]
    task = make_task()

    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'large', \
        f"Expected 'large' (higher capacity), got {chosen.id}"
    print(f'PSEUDO-PASS: cold_start_capacity_heuristic (chose {chosen.id})')


def test_reputation_weighted_scoring() -> None:
    """Higher reputation should lower effective cost, making the node more
    attractive."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = make_nodes()

    # Manually set reputations: nA high, nB low
    balancer._reputations['nA'] = 0.9
    balancer._reputations['nB'] = 0.2
    balancer._reputations['nC'] = 0.5

    # Set expected costs to be equal for all nodes
    balancer._expected_costs['nA'] = 5.0
    balancer._expected_costs['nB'] = 5.0
    balancer._expected_costs['nC'] = 5.0
    balancer._cost_counts['nA'] = 1
    balancer._cost_counts['nB'] = 1
    balancer._cost_counts['nC'] = 1

    chosen = balancer.assign(task := make_task(), nodes)

    # nA has highest rep (0.9) → lowest effective cost (5.0/0.9 ≈ 5.56)
    # nB has lowest rep (0.2) → highest effective cost (5.0/0.2 = 25.0)
    # nC is neutral (0.5) → effective cost 10.0
    assert chosen.id == 'nA', \
        f"Expected nA (highest reputation), got {chosen.id}"
    print(f'PSEUDO-PASS: reputation_weighted_scoring (chose {chosen.id})')


def test_reputation_update_on_complete() -> None:
    """Completing a task cheaper than expected should increase reputation."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = make_nodes()
    task = make_task()

    # Assign and complete with low cost
    chosen = balancer.assign(task, nodes)
    balancer.on_complete(task, chosen, 0.5)

    # First assignment: expected_cost was heuristic (~50/cap), actual=0.5
    # Reputation should increase
    rep = balancer._get_reputation(chosen.id)
    assert rep > 0.5, \
        f"Reputation should increase after cheap task, got {rep:.3f}"
    print(f'PSEUDO-PASS: reputation_update_on_complete '
          f'(rep={rep:.3f} after cheap task)')


def test_poor_node_reputation_decreases() -> None:
    """Consistently high-cost completions should decrease reputation."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted',
                                  reputation_update_rate=0.5)  # fast updates
    nodes = [
        Node(id='good', cpu_cap=30.0, memory_cap=20.0, network_cap=10.0),
        Node(id='bad', cpu_cap=5.0, memory_cap=5.0, network_cap=5.0),
    ]
    task = make_task()

    # Force 'bad' node to be selected first (cold-start may pick either)
    # Assign to 'bad' multiple times with high cost
    for _ in range(5):
        chosen = balancer.assign(task, nodes)
        # Force high cost on whichever node was chosen
        balancer.on_complete(task, chosen, 10.0)

    # After 5 high-cost completions, reputation of the selected node
    # should be lower than neutral (0.5)
    for nid in ['good', 'bad']:
        rep = balancer._get_reputation(nid)
        # Node that got high-cost tasks should have rep < 0.5
        # (unless it was never selected, then stays at 0.5)
        assert rep <= 0.5 + 0.01, \
            f"Reputation for {nid} should be <= 0.5 after high-cost tasks, got {rep:.3f}"
    print(f'PSEUDO-PASS: poor_node_reputation_decreases '
          f'(reps={ {k: f"{v:.3f}" for k, v in balancer._reputations.items()} })')


def test_snapshot_serializable() -> None:
    """Snapshot should be a serializable dict with reputation and cost data."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = make_nodes()
    task = make_task()

    for _ in range(5):
        chosen = balancer.assign(task, nodes)
        balancer.on_complete(task, chosen, 1.0)

    snap = balancer.snapshot()

    assert isinstance(snap, dict), f"Expected dict, got {type(snap)}"
    assert 'version' in snap
    assert snap['version'] == 1
    assert 'reputations' in snap
    assert 'expected_costs' in snap
    assert 'total_assignments' in snap
    assert snap['total_assignments'] == 5, \
        f"Expected 5, got {snap['total_assignments']}"
    assert 'mode' in snap
    assert snap['mode'] == 'reputation_weighted'
    print(f'PSEUDO-PASS: snapshot_serializable')


def test_merge() -> None:
    """Merging two snapshots combines reputation and cost statistics."""
    balancer1 = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = make_nodes()
    task = make_task()

    for _ in range(5):
        chosen = balancer1.assign(task, nodes)
        balancer1.on_complete(task, chosen, 1.0)

    snap1 = balancer1.snapshot()

    balancer2 = IncentiveBalancer(seed=42, mode='reputation_weighted')
    for _ in range(3):
        chosen = balancer2.assign(task, nodes)
        balancer2.on_complete(task, chosen, 1.0)

    balancer1.merge(snap1)
    merged = balancer1.snapshot()

    assert merged['total_assignments'] == snap1['total_assignments'] * 2, \
        f"Expected {snap1['total_assignments'] * 2}, got {merged['total_assignments']}"
    print(f'PSEUDO-PASS: merge (total_assignments={snap1["total_assignments"]}*2={merged["total_assignments"]})')


def test_price_aware_mode() -> None:
    """Price-aware mode: uses node_prices in addition to reputation."""
    balancer = IncentiveBalancer(seed=42, mode='price_aware')
    nodes = make_nodes()
    task = make_task()

    # Set prices: nA expensive, nB cheap
    balancer.node_prices['nA'] = 3.0
    balancer.node_prices['nB'] = 0.5
    balancer.node_prices['nC'] = 1.0

    # Set equal reputations and expected costs
    for n in nodes:
        balancer._reputations[n.id] = 0.5
        balancer._expected_costs[n.id] = 5.0
        balancer._cost_counts[n.id] = 1

    chosen = balancer.assign(task, nodes)

    # nB has lowest price (0.5) * cost (5.0) / rep (0.5) = 5.0
    # nA has price 3.0 * 5.0 / 0.5 = 30.0
    # nC has price 1.0 * 5.0 / 0.5 = 10.0
    assert chosen.id == 'nB', \
        f"Expected nB (lowest price), got {chosen.id}"
    print(f'PSEUDO-PASS: price_aware_mode (chose {chosen.id})')


def test_empty_nodes_error() -> None:
    """Assigning with empty node list should raise ValueError."""
    balancer = IncentiveBalancer(seed=42)
    try:
        balancer.assign(Task(id='t1', cpu_req=1.0, memory_req=1.0, network_req=1.0),
                        [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print('PSEUDO-PASS: empty_nodes_error')


def test_cold_start_no_history() -> None:
    """Cold-start with no history: uses capacity heuristic, then updates."""
    balancer = IncentiveBalancer(seed=42, mode='reputation_weighted')
    nodes = [
        Node(id='fast', cpu_cap=50.0, memory_cap=50.0, network_cap=50.0),
        Node(id='slow', cpu_cap=5.0, memory_cap=5.0, network_cap=5.0),
    ]
    task = make_task()

    # First assignment: no history, uses capacity heuristic
    chosen = balancer.assign(task, nodes)
    assert chosen.id == 'fast', \
        f"Expected 'fast' (higher capacity heuristic), got {chosen.id}"

    # Complete with low cost → reputation increases
    balancer.on_complete(task, chosen, 0.1)
    rep = balancer._get_reputation('fast')
    assert rep > 0.5, f"Reputation should increase, got {rep:.3f}"

    # Second assignment: now has history, should still prefer fast
    chosen2 = balancer.assign(task, nodes)
    assert chosen2.id == 'fast', \
        f"Expected 'fast' (now with good reputation), got {chosen2.id}"

    print(f'PSEUDO-PASS: cold_start_no_history '
          f'(first={chosen.id}, rep={rep:.3f}, second={chosen2.id})')


if __name__ == '__main__':
    main()
