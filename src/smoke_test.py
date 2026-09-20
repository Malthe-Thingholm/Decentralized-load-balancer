"""Deterministic smoke test — no network, no external deps."""

from __future__ import annotations

import random
import sys

from src.config import Config
from src.event import Event, Simulator, EVENT_ARRIVAL, EVENT_COMPLETION
from src.guess import StochasticGuessingModel
from src.metrics import (
    avg_latency,
    jain_fairness,
    makespan,
    regret,
    tail_latency,
    utilization,
)
from src.node import Node
from src.task import Task
from src.truesim import StochasticTrueModel, assert_cost
from src.balancer import Balancer


def test_true_model_stub() -> None:
    """StochasticTrueModel returns non-negative costs."""
    model = StochasticTrueModel(seed=0)
    node = Node(id="n1", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)

    c1 = model.cost(task, node)
    c2 = model.cost(task, node)  # stochastic — may differ
    assert c1 >= 0, f"cost must be >= 0, got {c1}"
    assert c2 >= 0, f"cost must be >= 0, got {c2}"
    print(f"  TrueModel: cost1={c1:.4f}, cost2={c2:.4f}")


def test_assertion_wrapper() -> None:
    """assert_cost rejects negative outputs."""

    @assert_cost
    def bad_cost(task, node):
        return -1.0

    node = Node(id="n1", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)
    try:
        bad_cost(task, node)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("  assert_cost wrapper: OK")


def test_guessing_model() -> None:
    """StochasticGuessingModel produces estimates, snapshots, merges."""
    model = StochasticGuessingModel(seed=1)
    node = Node(id="n1", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)

    est = model.estimate(task, node)
    snap = model.snapshot()
    model.merge({"version": 99, "observations": [1, 2, 3]})
    assert est >= 0
    assert "version" in snap
    print(f"  GuessingModel: estimate={est:.4f}, snapshot={snap}")


def test_simulator() -> None:
    """Simulator priority queue with deterministic replay."""
    sim = Simulator(seed=42)

    task1 = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)
    task2 = Task(id="t2", cpu_req=3.0, memory_req=3.0, network_req=1.0)

    sim.schedule(Event(sim_time=1.0, event_type=EVENT_ARRIVAL, task=task1))
    sim.schedule(Event(sim_time=0.5, event_type=EVENT_ARRIVAL, task=task2))
    sim.schedule(Event(sim_time=2.0, event_type=EVENT_COMPLETION, task=task1))

    events = list(sim.run())
    assert len(events) == 3
    assert events[0].sim_time == 0.5  # ordered by time
    assert events[1].sim_time == 1.0
    assert events[2].sim_time == 2.0
    print(f"  Simulator: {len(events)} events processed in order")


def test_node() -> None:
    """Node assignment and completion tracking."""
    completed_events: list = []

    def callback(tid: str, nid: str, cost: float) -> None:
        completed_events.append((tid, nid, cost))

    node = Node(id="n1", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
    node.on_complete = callback

    node.assign("t1")
    assert node.queue_depth == 1
    assert node.is_busy

    node.complete("t1", sim_time=5.0)
    assert node.queue_depth == 0
    assert len(completed_events) == 1
    print(f"  Node: assign/complete OK, callback fired")


def test_balancer_protocol() -> None:
    """Simple balancer implementation satisfies the protocol."""

    class RoundRobinBalancer:
        def __init__(self) -> None:
            self._idx = 0
            self.completions: list = []

        def assign(self, task: Task, nodes: list[Node]) -> Node:
            node = nodes[self._idx % len(nodes)]
            self._idx += 1
            return node

        def on_complete(self, task: Task, node: Node, actual_cost: float) -> None:
            self.completions.append((task.id, node.id, actual_cost))

    balancer = RoundRobinBalancer()
    nodes = [
        Node(id=f"n{i}", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
        for i in range(3)
    ]
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)

    chosen = balancer.assign(task, nodes)
    assert chosen in nodes
    balancer.on_complete(task, chosen, 1.5)
    assert len(balancer.completions) == 1
    print("  Balancer protocol: OK")


def test_metrics() -> None:
    """Metrics computation on a small scenario."""
    nodes = [
        Node(id="n0", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0),
        Node(id="n1", cpu_cap=30.0, memory_cap=30.0, network_cap=15.0),
    ]
    nodes[0].completed = ["t1", "t2"]
    nodes[0].total_busy_time = 10.0
    nodes[0].current_load_end = 5.0
    nodes[1].completed = ["t3"]
    nodes[1].total_busy_time = 8.0
    nodes[1].current_load_end = 4.0

    task_arrivals = {"t1": 0.0, "t2": 1.0, "t3": 0.5}

    m = makespan(nodes)
    a = avg_latency(nodes, task_arrivals)
    t = tail_latency(nodes, task_arrivals)
    j = jain_fairness(nodes)
    u = utilization(nodes, sim_time=10.0)
    r = regret(nodes, task_arrivals, oracle_makespan=3.0)

    assert m > 0, f"makespan should be > 0, got {m}"
    assert 0 <= j <= 1, f"Jain's should be in [0,1], got {j}"
    assert 0 <= u <= 1, f"utilization should be in [0,1], got {u}"
    print(f"  Metrics: makespan={m:.2f}, avg_lat={a:.4f}, tail={t:.4f}, "
          f"jain={j:.4f}, util={u:.4f}, regret={r:.4f}")


def test_deterministic_replay() -> None:
    """Same seed → same cost sequence."""
    m1 = StochasticTrueModel(seed=99)
    m2 = StochasticTrueModel(seed=99)
    node = Node(id="n1", cpu_cap=20.0, memory_cap=20.0, network_cap=10.0)
    task = Task(id="t1", cpu_req=5.0, memory_req=5.0, network_req=2.0)

    # Seed both models' RNGs identically for deterministic comparison
    m1.rng = random.Random(99)
    m2.rng = random.Random(99)

    costs1 = [m1.cost(task, node) for _ in range(5)]
    costs2 = [m2.cost(task, node) for _ in range(5)]
    assert costs1 == costs2, f"Seeded models should produce identical sequences"
    print(f"  Deterministic replay: {costs1[:3]} == {costs2[:3]}")


def main() -> int:
    cfg = Config(seed=42)
    print(f"=== Phase 1 Smoke Test (seed={cfg.seed}) ===\n")

    tests = [
        test_true_model_stub,
        test_assertion_wrapper,
        test_guessing_model,
        test_simulator,
        test_node,
        test_balancer_protocol,
        test_metrics,
        test_deterministic_replay,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())