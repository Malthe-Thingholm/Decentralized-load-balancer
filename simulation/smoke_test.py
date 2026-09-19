"""Smoke test — proves Phase 1 core loop works end-to-end."""

from __future__ import annotations

from simulation.engine import Engine
from simulation.true_model import AssertedTrueModel, TrueModel
from simulation.types import (
    Assignment,
    Capability,
    Event,
    Task,
    EVENT_ARRIVAL,
    EVENT_COMPLETION,
)


def _on_arrival(event: Event, engine: Engine) -> None:
    assert event.task is not None
    task = event.task
    best_node = min(engine.nodes, key=lambda nid: engine.nodes[nid].current_load)
    node = engine.nodes[best_node]
    cost = engine.true_model.cost(task, node.capability)  # type: ignore[attr-defined]
    assignment = Assignment(task=task, node_id=best_node, cost=cost)
    node.queue.append(assignment)
    node.current_load += cost
    engine.schedule(
        Event(
            sim_time=engine.time + cost,
            event_type=EVENT_COMPLETION,
            task=task,
            node_id=best_node,
        )
    )


def _on_completion(event: Event, engine: Engine) -> None:
    assert event.task is not None and event.node_id is not None
    node = engine.nodes[event.node_id]
    assignment = node.queue.pop(0)
    node.completed.append(assignment)
    node.total_busy_time += assignment.cost or 0.0
    node.current_load = max(0.0, node.current_load - (assignment.cost or 0.0))


def smoke_test() -> None:
    """End-to-end smoke test: 3 tasks, 2 nodes, all complete."""
    engine = Engine(seed=123)
    true_model = TrueModel(seed=123)
    engine.true_model = AssertedTrueModel(true_model)  # type: ignore[attr-defined]

    engine.add_node("n1", Capability(cpu=4.0, memory=8.0, network=1.0))
    engine.add_node("n2", Capability(cpu=2.0, memory=4.0, network=1.0))

    engine.on(EVENT_ARRIVAL, _on_arrival)
    engine.on(EVENT_COMPLETION, _on_completion)

    tasks = [
        Task(id="t1", cpu_req=1.0, memory_req=2.0, network_req=0.5),
        Task(id="t2", cpu_req=2.0, memory_req=1.0, network_req=0.3),
        Task(id="t3", cpu_req=1.5, memory_req=3.0, network_req=0.4),
    ]

    for t in tasks:
        engine.schedule(
            Event(sim_time=0.0, event_type=EVENT_ARRIVAL, task=t)
        )

    engine.run()

    total_completed = sum(len(s.completed) for s in engine.nodes.values())
    assert total_completed == 3, f"Expected 3 completions, got {total_completed}"

    engine.true_model.assert_consumed([t.id for t in tasks])  # type: ignore[union-attr]

    snap = engine.snapshot()
    assert snap["time"] > 0.0
    assert snap["queue_len"] == 0

    print(f"PASS — {total_completed} tasks completed across {len(engine.nodes)} nodes")
    print(f"  Final time: {snap['time']:.2f}")
    for nid, ns in snap["nodes"].items():
        print(f"  {nid}: completed={ns['completed']} busy={ns['total_busy_time']:.2f}")


if __name__ == "__main__":
    smoke_test()