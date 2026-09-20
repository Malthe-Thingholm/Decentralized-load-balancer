"""Pseudo-tests for decentralized load balancing simulation with gossip.

Validates: node creation, gossip mechanics, simulation flow, convergence
measurement, and compare_decentralized_balancers.
"""

from __future__ import annotations

from src.decentralized_simulator import (
    DecentralizedConfig,
    DecentralizedNode,
    DecentralizedResult,
    DecentralizedSimulation,
    compare_decentralized_balancers,
    create_heterogeneous_nodes,
    gossip_broadcast,
    gossip_pair,
)
from src.random_balancer import RandomBalancer
from src.shortest_queue_balancer import ShortestQueueBalancer
from src.node import Node


def main() -> None:
    tests = [
        test_create_heterogeneous_nodes,
        test_gossip_pair_no_gossip_support,
        test_gossip_pair_with_gossip,
        test_gossip_broadcast_skips_non_gossip,
        test_simulation_basic_run,
        test_simulation_all_same_balancer_type,
        test_compare_decentralized_balancers,
        test_convergence_no_gossip_capable,
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

    print(f'\n=== Decentralized Simulator Pseudo-Test Results ===\n\nPseudo-test results: {passed} passed, {failed} failed')
    if failed:
        raise SystemExit(1)


def test_create_heterogeneous_nodes() -> None:
    """Heterogeneous nodes should have different random capacities."""
    rng = __import__('random').Random(42)
    nodes = create_heterogeneous_nodes(5, rng, cpu_mean=3.0, cpu_std=0.5)
    assert len(nodes) == 5
    caps = [n.cpu_cap for n in nodes]
    assert len(set(round(c, 2) for c in caps)) >= 3, \
        f"Expected diverse capacities, got {caps}"
    print(f'PSEUDO-PASS: create_heterogeneous_nodes (caps={[round(c,1) for c in caps]})')


def test_gossip_pair_no_gossip_support() -> None:
    """Non-gossip-capable balancers (RandomBalancer) should be skipped."""
    n1 = DecentralizedNode(
        node_id='n1',
        node=Node(id='node-1', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=RandomBalancer(),
    )
    n2 = DecentralizedNode(
        node_id='n2',
        node=Node(id='node-2', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=RandomBalancer(),
    )
    result = gossip_pair(n1, n2)
    assert result is None, f"Expected None for non-gossip balancers, got {result}"
    print('PSEUDO-PASS: gossip_pair_no_gossip_support (skipped)')


def test_gossip_pair_with_gossip() -> None:
    """Gossip-capable balancers should exchange state."""
    from src.mf_mab_balancer import MeanFieldMABBalancer

    n1 = DecentralizedNode(
        node_id='n1',
        node=Node(id='node-1', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=MeanFieldMABBalancer(seed=1),
    )
    n2 = DecentralizedNode(
        node_id='n2',
        node=Node(id='node-2', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=MeanFieldMABBalancer(seed=2),
    )

    # Pre-populate state
    from src.task import Task
    for _ in range(3):
        t = Task(id='t', cpu_req=5.0, memory_req=5.0, network_req=5.0)
        n1.balancer.assign(t, [n1.node, n2.node])

    result = gossip_pair(n1, n2)
    assert result is not None, "Expected gossip to succeed"
    assert result.successful is True
    print(f'PSEUDO-PASS: gossip_pair_with_gossip (successful={result.successful})')


def test_gossip_broadcast_skips_non_gossip() -> None:
    """broadcast should skip non-gossip-capable peers."""
    from src.mf_mab_balancer import MeanFieldMABBalancer

    source = DecentralizedNode(
        node_id='src',
        node=Node(id='node-src', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=MeanFieldMABBalancer(seed=42),
    )
    peer_no_gossip = DecentralizedNode(
        node_id='peer1',
        node=Node(id='node-p1', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=RandomBalancer(),
    )
    peer_with_gossip = DecentralizedNode(
        node_id='peer2',
        node=Node(id='node-p2', cpu_cap=10.0, memory_cap=10.0, network_cap=10.0),
        balancer=MeanFieldMABBalancer(seed=99),
    )

    exchanges = gossip_broadcast(source, [peer_no_gossip, peer_with_gossip])
    assert len(exchanges) == 1, f"Expected 1 exchange (only gossip peer), got {len(exchanges)}"
    assert exchanges[0].receiver_id == 'peer2'
    print(f'PSEUDO-PASS: gossip_broadcast_skips_non_gossip (exchanges={len(exchanges)})')


def test_simulation_basic_run() -> None:
    """A basic simulation should produce valid results."""
    cfg = DecentralizedConfig(num_nodes=3, sim_duration=20.0, seed=42)
    sim = DecentralizedSimulation(
        balancer_class=RandomBalancer, config=cfg,
    )
    result = sim.run()

    assert result.makespan > 0, f"Expected positive makespan, got {result.makespan}"
    assert result.tasks_generated > 0, f"Expected tasks generated, got {result.tasks_generated}"
    assert result.balancer_type == 'RandomBalancer'
    assert result.state_convergence is None  # RandomBalancer not gossip-capable
    print(f'PSEUDO-PASS: simulation_basic_run '
          f'(makespan={result.makespan:.2f}, tasks={result.tasks_completed}/{result.tasks_generated})')


def test_simulation_all_same_balancer_type() -> None:
    """All nodes in a run must use the same balancer type."""
    cfg = DecentralizedConfig(num_nodes=4, sim_duration=10.0, seed=42)
    sim = DecentralizedSimulation(
        balancer_class=ShortestQueueBalancer, config=cfg,
    )
    sim.run()

    types = set(type(n.balancer).__name__ for n in sim._nodes)
    assert types == {'ShortestQueueBalancer'}, \
        f"Expected all ShortestQueueBalancer, got {types}"
    print(f'PSEUDO-PASS: simulation_all_same_balancer_type (types={types})')


def test_compare_decentralized_balancers() -> None:
    """Comparing two balancer types should return results for each."""
    cfg = DecentralizedConfig(num_nodes=3, sim_duration=15.0, seed=42, gossip_interval=999.0)
    results = compare_decentralized_balancers(
        [RandomBalancer, ShortestQueueBalancer],
        config=cfg,
        num_seeds=1,
    )
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].balancer_type == 'RandomBalancer'
    assert results[1].balancer_type == 'ShortestQueueBalancer'
    assert all(r.makespan > 0 for r in results)
    print(f'PSEUDO-PASS: compare_decentralized_balancers '
          f'(Random={results[0].makespan:.2f}, SQ={results[1].makespan:.2f})')


def test_convergence_no_gossip_capable() -> None:
    """Non-gossip-capable balancers should return None for convergence."""
    cfg = DecentralizedConfig(num_nodes=3, sim_duration=10.0, seed=42)
    sim = DecentralizedSimulation(
        balancer_class=RandomBalancer, config=cfg,
    )
    result = sim.run()
    assert result.state_convergence is None
    print(f'PSEUDO-PASS: convergence_no_gossip_capable (convergence={result.state_convergence})')


if __name__ == '__main__':
    main()
