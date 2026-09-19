"""Phase 1 smoke test — exercises engine + TrueModel + assertions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.engine import Engine, EventType
from sim.true_model import TrueModel, Task, Node
from sim.assertions import AssertionWrapper


def test_engine_runs() -> None:
    """Engine processes scheduled events in time order."""
    eng = Engine(seed=1)
    order: list[str] = []

    def tag(name: str) -> None:
        order.append(name)

    eng.on(EventType.ARRIVAL, lambda e: tag("arrival"))
    eng.on(EventType.COMPLETION, lambda e: tag("completion"))
    eng.schedule(EventType.ARRIVAL, delay=2.0)
    eng.schedule(EventType.COMPLETION, delay=1.0)
    eng.schedule(EventType.ARRIVAL, delay=3.0)
    eng.run()

    assert order == ["completion", "arrival", "arrival"], f"got {order}"
    assert eng.time == 3.0
    print("  PASS: engine_runs")


def test_true_model_deterministic() -> None:
    """Same seed gives same costs."""
    m1 = TrueModel(seed=7)
    m2 = TrueModel(seed=7)
    t = Task(requirements={"cpu": 2.0, "mem": 1.0}, task_id="t1")
    n = Node(capability={"cpu": 4.0, "mem": 4.0}, node_id="n1")
    assert m1.cost(t, n) == m2.cost(t, n)
    print("  PASS: true_model_deterministic")


def test_true_model_stochastic_across_seeds() -> None:
    """Different seeds give different costs (high probability)."""
    m1 = TrueModel(seed=1)
    m2 = TrueModel(seed=2)
    t = Task(requirements={"cpu": 2.0}, task_id="t1")
    n = Node(capability={"cpu": 4.0}, node_id="n1")
    c1 = m1.cost(t, n)
    c2 = m2.cost(t, n)
    assert c1 != c2, f"same cost {c1} across seeds"
    print("  PASS: true_model_stochastic_across_seeds")


def test_assertion_wrapper_passes() -> None:
    """Valid costs pass the assertion wrapper."""
    model = TrueModel(seed=0)
    aw = AssertionWrapper(model)
    t = Task(requirements={"cpu": 1.0}, task_id="t1")
    n = Node(capability={"cpu": 2.0}, node_id="n1")
    c = aw.cost(t, n)
    assert c > 0
    aw.assert_clean()
    print("  PASS: assertion_wrapper_passes")


def test_assertion_wrapper_rejects_negative() -> None:
    """Wrapper raises on negative cost."""
    class BadModel(TrueModel):
        def cost(self, task, node):
            return -1.0

    aw = AssertionWrapper(BadModel())
    t = Task(requirements={"cpu": 1.0}, task_id="t1")
    n = Node(capability={"cpu": 2.0}, node_id="n1")
    try:
        aw.cost(t, n)
        assert False, "should have raised"
    except AssertionError:
        pass
    # assert_clean() raises because errors accumulated — expected here
    try:
        aw.assert_clean()
        assert False, "assert_clean should have raised"
    except AssertionError:
        pass
    print("  PASS: assertion_wrapper_rejects_negative")


def test_engine_snapshot() -> None:
    eng = Engine(seed=5)
    snap = eng.snapshot()
    assert snap["time"] == 0.0
    assert snap["step_count"] == 0
    assert snap["seed"] == 5
    print("  PASS: engine_snapshot")


def main() -> None:
    tests = [
        test_engine_runs,
        test_true_model_deterministic,
        test_true_model_stochastic_across_seeds,
        test_assertion_wrapper_passes,
        test_assertion_wrapper_rejects_negative,
        test_engine_snapshot,
    ]
    passed = 0
    for t in tests:
        print(f"Running {t.__name__} ...")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()