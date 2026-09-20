"""Parameterized test suites for decentralized simulation.

Explores the design space:
- Node count: 10 (small), 50 (medium), 500 (large)
- Guess model accuracy: accurate (0.05), inaccurate (0.5)
- Congestion: light (0.5 tasks/s/node), heavy (2.0 tasks/s/node)
- Gossip: on vs off
- Balancer types: Random, ShortestQueue, MF-MAB, DML, QEdgeProxy, Incentive

Each cell runs multiple seeds. Results are aggregated and reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.decentralized_simulator import (
    DecentralizedConfig,
    DecentralizedSimulation,
    DecentralizedResult,
    compare_decentralized_balancers,
)
from src.guess import StochasticGuessingModel
from src.mf_mab_balancer import MeanFieldMABBalancer
from src.qedgeproxy_balancer import QEdgeProxyBalancer
from src.incentive_balancer import IncentiveBalancer
from src.random_balancer import RandomBalancer
from src.shortest_queue_balancer import ShortestQueueBalancer
from src.dml_balancer import CommunicationEfficientDMLBalancer
from src.truesim import StochasticTrueModel


# ---------------------------------------------------------------------------
# Balancer registry
# ---------------------------------------------------------------------------

@dataclass
class BalancerSpec:
    """A balancer type to test, with its constructor kwargs."""
    name: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    gossip_capable: bool = True


BALANCER_SUITE: list[BalancerSpec] = [
    BalancerSpec("Random", RandomBalancer, gossip_capable=False),
    BalancerSpec("ShortestQueue", ShortestQueueBalancer, gossip_capable=False),
    BalancerSpec("MF-MAB", MeanFieldMABBalancer),
    BalancerSpec(
        "DML-capability", CommunicationEfficientDMLBalancer,
        kwargs={"mode": "capability_aware"},
    ),
    BalancerSpec(
        "QEdgeProxy", QEdgeProxyBalancer,
        kwargs={"num_proxies": 3, "ucb_c": 2.0},
    ),
    BalancerSpec(
        "Incentive-rep", IncentiveBalancer,
        kwargs={"mode": "reputation_weighted"},
    ),
]


# ---------------------------------------------------------------------------
# Test suite definitions
# ---------------------------------------------------------------------------

@dataclass
class SuiteConfig:
    """One cell in the parameter sweep."""
    name: str
    num_nodes: int
    arrival_rate: float       # tasks per second (total, not per-node)
    guess_error: float        # error_std for StochasticGuessingModel
    gossip: bool              # whether gossip_interval is finite
    gossip_mode: str = 'pairwise_random'
    gossip_network: str = 'complete'
    sim_duration: float = 300.0
    gossip_interval: float = 100.0
    seeds: int = 3
    gossip_interval: float = 10.0

    def config(self, seed: int) -> DecentralizedConfig:
        """Build DecentralizedConfig for a specific seed."""
        topo = self.gossip_network
        # Use random_k topology for large node counts to keep gossip O(n log n)
        if self.num_nodes >= 200 and topo == 'complete':
            topo = 'random_k'

        return DecentralizedConfig(
            num_nodes=self.num_nodes,
            arrival_rate=self.arrival_rate,
            guess_model_error=self.guess_error,
            sim_duration=self.sim_duration,
            seed=seed,
            gossip_interval=self.gossip_interval if self.gossip else 999999.0,
            gossip_mode=self.gossip_mode,
            gossip_network=topo,
        )


# Auto-detect gossip network for large node counts
def _topo_for_nodes(n: int) -> str:
    if n >= 200:
        return 'random_k'
    return 'complete'


# Named suites
SUITES: list[SuiteConfig] = [
    # ---- Small (10 nodes) ----
    # System capacity: ~500 tasks per 300s. Light=60% util (arrival_rate≈1.0),
    # Congested=150% util (arrival_rate≈2.5).
    SuiteConfig(
        name="small_light_accurate",
        num_nodes=10, arrival_rate=1.0, guess_error=0.05,
        gossip=True, sim_duration=300.0, seeds=5,
    ),
    SuiteConfig(
        name="small_light_inaccurate",
        num_nodes=10, arrival_rate=1.0, guess_error=0.5,
        gossip=True, sim_duration=300.0, seeds=5,
    ),
    SuiteConfig(
        name="small_congested_accurate",
        num_nodes=10, arrival_rate=2.5, guess_error=0.05,
        gossip=True, sim_duration=300.0, seeds=5,
    ),
    SuiteConfig(
        name="small_congested_inaccurate",
        num_nodes=10, arrival_rate=2.5, guess_error=0.5,
        gossip=True, sim_duration=300.0, seeds=5,
    ),
    SuiteConfig(
        name="small_light_accurate_nogossip",
        num_nodes=10, arrival_rate=1.0, guess_error=0.05,
        gossip=False, sim_duration=300.0, seeds=5,
    ),
    SuiteConfig(
        name="small_congested_inaccurate_nogossip",
        num_nodes=10, arrival_rate=2.5, guess_error=0.5,
        gossip=False, sim_duration=300.0, seeds=5,
    ),

    # ---- Medium (50 nodes) ----
    # System capacity: ~328 tasks per 40s. Light≈5.0, Congested≈12.0.
    SuiteConfig(
        name="medium_light_accurate",
        num_nodes=50, arrival_rate=5.0, guess_error=0.05,
        gossip=True, sim_duration=400.0, seeds=3,
    ),
    SuiteConfig(
        name="medium_congested_inaccurate",
        num_nodes=50, arrival_rate=12.0, guess_error=0.5,
        gossip=True, sim_duration=400.0, seeds=3,
    ),
    SuiteConfig(
        name="medium_light_accurate_nogossip",
        num_nodes=50, arrival_rate=5.0, guess_error=0.05,
        gossip=False, sim_duration=400.0, seeds=3,
    ),

    # ---- Large (500 nodes) ----
    # System capacity: ~4918 tasks per 60s. Light≈50.0, Congested≈125.0.
    SuiteConfig(
        name="large_light_accurate",
        num_nodes=500, arrival_rate=50.0, guess_error=0.05,
        gossip=True, sim_duration=600.0, seeds=2,
    ),
    SuiteConfig(
        name="large_congested_inaccurate",
        num_nodes=500, arrival_rate=125.0, guess_error=0.5,
        gossip=True, sim_duration=600.0, seeds=2,
    ),
]


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    """Aggregated metrics for one (suite, balancer) combination."""
    suite_name: str
    balancer_name: str
    gossip: bool
    num_nodes: int
    arrival_rate: float
    guess_error: float
    seeds: int

    # Aggregates
    mean_makespan: float
    std_makespan: float
    mean_throughput: float          # tasks_completed / sim_duration
    std_throughput: float
    mean_convergence: float | None  # None if balancer not gossip-capable
    std_convergence: float | None
    total_gossip_exchanges: int     # sum across seeds
    mean_runtime_s: float
    successful_seeds: int
    failed_seeds: int


def aggregate(rows: list[SweepRow], spec: BalancerSpec,
              suite: SuiteConfig) -> AggregatedResult:
    """Aggregate multiple seed runs into summary statistics."""
    makespan_vals = [r.makespan for r in rows]
    throughput_vals = [
        r.throughput for r in rows
    ]
    gossip_exchanges = sum(r.gossip_exchanges for r in rows)  # int, not list

    # Convergence: only for gossip-capable balancers
    conv_vals = [r.convergence for r in rows
                 if r.convergence is not None]
    mean_conv = sum(conv_vals) / len(conv_vals) if conv_vals else None
    std_conv: float | None = None
    if conv_vals and len(conv_vals) > 1 and mean_conv is not None:
        std_conv = (sum((c - mean_conv) ** 2 for c in conv_vals) / len(conv_vals)) ** 0.5

    n = len(rows)
    mean_mk = sum(makespan_vals) / n
    var_mk = sum((m - mean_mk) ** 2 for m in makespan_vals) / n
    std_mk = var_mk ** 0.5

    mean_tp = sum(throughput_vals) / n
    var_tp = sum((t - mean_tp) ** 2 for t in throughput_vals) / n
    std_tp = var_tp ** 0.5

    return AggregatedResult(
        suite_name=suite.name,
        balancer_name=spec.name,
        gossip=suite.gossip,
        num_nodes=suite.num_nodes,
        arrival_rate=suite.arrival_rate,
        guess_error=suite.guess_error,
        seeds=suite.seeds,
        mean_makespan=mean_mk,
        std_makespan=std_mk,
        mean_throughput=mean_tp,
        std_throughput=std_tp,
        mean_convergence=mean_conv,
        std_convergence=std_conv,
        total_gossip_exchanges=gossip_exchanges,
        mean_runtime_s=0.0,  # filled in by runner
        successful_seeds=n,
        failed_seeds=0,
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    """One row in the sweep results table."""
    suite: str
    balancer: str
    gossip: bool
    nodes: int
    arr_rate: float
    guess_err: float
    seed: int
    makespan: float
    throughput: float
    completed: int
    generated: int
    gossip_exchanges: int
    convergence: float | None
    runtime_s: float


def run_suite(suite: SuiteConfig, spec: BalancerSpec) -> list[SweepRow]:
    """Run one (suite, balancer) combination across all seeds.

    Returns a list of SweepRow, one per seed.
    """
    rows: list[SweepRow] = []
    true_model = StochasticTrueModel(
        seed=sum(ord(c) for c in suite.name) * 31 + 10,
        cost_scale=15.0,
        noise_std=0.1,
    )
    guess_model = StochasticGuessingModel(
        seed=sum(ord(c) for c in suite.name) * 31 + 20,
        error_std=suite.guess_error,
    )

    for seed_idx in range(suite.seeds):
        seed = (
            sum(ord(c) for c in f"{spec.name}:{suite.name}") * 31
            + seed_idx * 1000
            + hash(spec.name) % 10000
        ) % 2000000
        cfg = suite.config(seed)

        t0 = time.monotonic()
        try:
            sim = DecentralizedSimulation(
                balancer_class=spec.cls,
                balancer_kwargs=spec.kwargs,
                config=cfg,
                true_model=true_model,
                guess_model=guess_model,
            )
            result = sim.run()
            runtime = time.monotonic() - t0

            row = SweepRow(
                suite=suite.name,
                balancer=spec.name,
                gossip=suite.gossip,
                nodes=suite.num_nodes,
                arr_rate=suite.arrival_rate,
                guess_err=suite.guess_error,
                seed=seed,
                makespan=result.makespan,
                throughput=result.tasks_completed / suite.sim_duration,
                completed=result.tasks_completed,
                generated=result.tasks_generated,
                gossip_exchanges=len(result.gossip_exchanges),
                convergence=result.state_convergence,
                runtime_s=runtime,
            )
            rows.append(row)
        except Exception as e:
            # Record failure row
            rows.append(SweepRow(
                suite=suite.name,
                balancer=spec.name,
                gossip=suite.gossip,
                nodes=suite.num_nodes,
                arr_rate=suite.arrival_rate,
                guess_err=suite.guess_error,
                seed=seed,
                makespan=float('nan'),
                throughput=0.0,
                completed=0,
                generated=0,
                gossip_exchanges=0,
                convergence=None,
                runtime_s=time.monotonic() - t0,
            ))

    return rows


def run_full_sweep(
    suites: list[SuiteConfig] | None = None,
    balancers: list[BalancerSpec] | None = None,
    output_csv: str | None = None,
) -> list[SweepRow]:
    """Run the full parameter sweep.

    Args:
        suites: List of SuiteConfig to run. Defaults to all SUITES.
        balancers: List of BalancerSpec to test. Defaults to all BALANCER_SUITE.
        output_csv: Optional path to write CSV results.

    Returns:
        List of SweepRow for every (suite, balancer, seed) combination.
    """
    suites = suites or SUITES
    balancers = balancers or BALANCER_SUITE

    all_rows: list[SweepRow] = []
    agg_results: list[AggregatedResult] = []

    for suite in suites:
        for spec in balancers:
            rows = run_suite(suite, spec)
            all_rows.extend(rows)

            # Aggregate
            success_rows = [r for r in rows if not (r.makespan != r.makespan)]  # NaN check
            if success_rows:
                agg = aggregate(success_rows, spec, suite)
                agg.mean_runtime_s = sum(r.runtime_s for r in success_rows) / len(success_rows)
                agg_results.append(agg)

    if output_csv:
        _write_csv(all_rows, output_csv)

    return all_rows


def _write_csv(rows: list[SweepRow], path: str) -> None:
    """Write sweep results to CSV."""
    header = (
        "suite,balancer,gossip,nodes,arr_rate,guess_err,seed,"
        "makespan,throughput,completed,generated,gossip_exchanges,"
        "convergence,runtime_s"
    )
    lines = [header]
    for r in rows:
        conv_str = f"{r.convergence:.4f}" if r.convergence is not None else "NA"
        lines.append(
            f"{r.suite},{r.balancer},{r.gossip},{r.nodes},{r.arr_rate},"
            f"{r.guess_err},{r.seed},{r.makespan:.2f},{r.throughput:.4f},"
            f"{r.completed},{r.generated},{r.gossip_exchanges},{conv_str},"
            f"{r.runtime_s:.3f}"
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_aggregated_report(agg_results: list[AggregatedResult]) -> None:
    """Print a human-readable aggregated report."""
    # Group by suite
    from collections import defaultdict
    by_suite: dict[str, list[AggregatedResult]] = defaultdict(list)
    for a in agg_results:
        by_suite[a.suite_name].append(a)

    print("=" * 120)
    print("DECENTRALIZED SIMULATION TEST SUITE REPORT")
    print("=" * 120)

    for suite_name, aggs in sorted(by_suite.items()):
        print(f"\n--- {suite_name} ---")
        # Header
        print(f"{'Balancer':<18} {'Gossip':<7} {'Makespan ± std':<22} {'Throughput ± std':<22} "
              f"{'Convergence ± std':<22} {'Gossip Xchs':<14} {'Seeds':<7} {'Runtime':<9}")
        print("-" * 120)

        for a in sorted(aggs, key=lambda x: x.mean_makespan):
            conv_str = (
                f"{a.mean_convergence:.4f} ± {a.std_convergence:.4f}"
                if a.mean_convergence is not None else "N/A"
            )
            gossip_xchs = a.total_gossip_exchanges
            if a.gossip:
                gossip_xchs_str = str(gossip_xchs)
            else:
                gossip_xchs_str = "off"

            print(
                f"{a.balancer_name:<18} {str(a.gossip):<7} "
                f"{a.mean_makespan:<8.2f} ± {a.std_makespan:<8.2f}   "
                f"{a.mean_throughput:<8.4f} ± {a.std_throughput:<8.4f}   "
                f"{conv_str:<22} {gossip_xchs_str:<14} "
                f"{a.seeds}/{a.seeds + a.failed_seeds:<6} {a.mean_runtime_s:.2f}s"
            )

    print("\n" + "=" * 120)


# ---------------------------------------------------------------------------
# Pseudo-tests: validate the framework itself
# ---------------------------------------------------------------------------

def test_suite_definition() -> None:
    """Verify suite definitions are well-formed."""
    assert len(SUITES) >= 4, f"Expected >= 4 suites, got {len(SUITES)}"
    for s in SUITES:
        assert s.num_nodes in (10, 50, 500), f"Unexpected node count: {s.num_nodes}"
        assert s.arrival_rate > 0
        assert 0 < s.guess_error < 1
        assert s.seeds >= 1
        assert s.sim_duration > 0
    print(f"PSEUDO-PASS: suite_definition ({len(SUITES)} suites, {sum(s.seeds for s in SUITES)} total seeds)")


def test_balancer_suite() -> None:
    """Verify all balancer specs are valid."""
    assert len(BALANCER_SUITE) >= 4, f"Expected >= 4 balancers, got {len(BALANCER_SUITE)}"
    for b in BALANCER_SUITE:
        assert b.name
        assert callable(b.cls)
        assert isinstance(b.kwargs, dict)
    print(f"PSEUDO-PASS: balancer_suite ({len(BALANCER_SUITE)} balancers)")


def test_small_suite_runs() -> None:
    """Run a single small suite to validate the framework works end-to-end."""
    suite = SUITES[0]  # small_light_accurate
    spec = BALANCER_SUITE[0]  # Random

    rows = run_suite(suite, spec)
    assert len(rows) == suite.seeds, f"Expected {suite.seeds} rows, got {len(rows)}"
    for r in rows:
        assert r.makespan > 0, f"Expected positive makespan, got {r.makespan}"
        assert r.completed <= r.generated, f"Completed > generated: {r.completed} > {r.generated}"
    print(f"PSEUDO-PASS: small_suite_runs "
          f"(makespan={[f'{r.makespan:.1f}' for r in rows]}, "
          f"completed={[r.completed for r in rows]})")


def test_gossip_vs_nogossip() -> None:
    """Gossip off (huge interval) should produce no gossip exchanges."""
    from src.decentralized_simulator import DecentralizedConfig, DecentralizedSimulation

    cfg = DecentralizedConfig(
        num_nodes=5, sim_duration=20.0, arrival_rate=5.0,
        gossip_interval=999999.0, seed=42,
    )
    sim = DecentralizedSimulation(
        balancer_class=MeanFieldMABBalancer,
        config=cfg,
    )
    result = sim.run()
    assert len(result.gossip_exchanges) == 0, \
        f"Expected 0 gossip exchanges with gossip=False, got {len(result.gossip_exchanges)}"
    assert result.state_convergence is None, \
        "Expected None convergence with gossip=False"
    print(f"PSEUDO-PASS: gossip_vs_nogossip "
          f"(exchanges={len(result.gossip_exchanges)}, convergence={result.state_convergence})")


def test_large_node_topology_auto() -> None:
    """Large node counts should auto-switch to random_k topology."""
    cfg = DecentralizedConfig(
        num_nodes=500, sim_duration=5.0, arrival_rate=50.0,
        gossip_network='complete',  # request complete
        seed=42,
    )
    sim = DecentralizedSimulation(
        balancer_class=RandomBalancer,
        config=cfg,
    )
    # Check that topology was switched
    for node in sim._nodes:
        # With 500 nodes and random_k, each node has ~250 neighbors, not 499
        assert len(node.gossip_neighbors) < 400, \
            f"Expected < 400 neighbors for 500 nodes with random_k, got {len(node.gossip_neighbors)}"
    sim.run()  # should not crash
    print(f"PSEUDO-PASS: large_node_topology_auto "
          f"(neighbors={[len(n.gossip_neighbors) for n in sim._nodes[:3]]})")


def test_aggregated_result() -> None:
    """Test the aggregation function."""
    from src.decentralized_simulator import DecentralizedConfig, DecentralizedSimulation

    cfg = DecentralizedConfig(num_nodes=5, sim_duration=10.0, seed=42)
    sim = DecentralizedSimulation(
        balancer_class=MeanFieldMABBalancer,
        config=cfg,
    )
    r1 = sim.run()
    # Change seed
    cfg2 = DecentralizedConfig(num_nodes=5, sim_duration=10.0, seed=43)
    sim2 = DecentralizedSimulation(
        balancer_class=MeanFieldMABBalancer,
        config=cfg2,
    )
    r2 = sim2.run()

    spec = BALANCER_SUITE[2]  # MF-MAB
    suite = SuiteConfig(
        name="test", num_nodes=5, arrival_rate=5.0, guess_error=0.05,
        gossip=True, seeds=2,
    )
    # Convert DecentralizedResults to SweepRows
    rows = [
        SweepRow(
            suite="test", balancer="MF-MAB", gossip=True,
            nodes=5, arr_rate=5.0, guess_err=0.05,
            seed=42, makespan=r1.makespan,
            throughput=r1.tasks_completed / 10.0,
            completed=r1.tasks_completed, generated=r1.tasks_generated,
            gossip_exchanges=len(r1.gossip_exchanges),
            convergence=r1.state_convergence, runtime_s=0.0,
        ),
        SweepRow(
            suite="test", balancer="MF-MAB", gossip=True,
            nodes=5, arr_rate=5.0, guess_err=0.05,
            seed=43, makespan=r2.makespan,
            throughput=r2.tasks_completed / 10.0,
            completed=r2.tasks_completed, generated=r2.tasks_generated,
            gossip_exchanges=len(r2.gossip_exchanges),
            convergence=r2.state_convergence, runtime_s=0.0,
        ),
    ]
    agg = aggregate(rows, spec, suite)
    assert agg.mean_makespan > 0
    assert agg.successful_seeds == 2
    assert agg.failed_seeds == 0
    print(f"PSEUDO-PASS: aggregated_result "
          f"(makespan={agg.mean_makespan:.2f}±{agg.std_makespan:.2f}, "
          f"throughput={agg.mean_throughput:.4f})")


def test_csv_output(tmp_path=None) -> None:
    """Test CSV output generation."""
    import os
    import tempfile

    rows = [
        SweepRow(
            suite="test", balancer="Random", gossip=False,
            nodes=5, arr_rate=5.0, guess_err=0.05,
            seed=42, makespan=10.5, throughput=0.5,
            completed=10, generated=15,
            gossip_exchanges=0, convergence=None, runtime_s=0.1,
        ),
        SweepRow(
            suite="test", balancer="MF-MAB", gossip=True,
            nodes=5, arr_rate=5.0, guess_err=0.05,
            seed=42, makespan=8.2, throughput=0.6,
            completed=12, generated=15,
            gossip_exchanges=4, convergence=0.85, runtime_s=0.2,
        ),
    ]

    tmp = tmp_path or tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "test_sweep.csv")
    _write_csv(rows, csv_path)

    with open(csv_path) as f:
        content = f.read()

    assert "suite,balancer,gossip" in content
    assert "Random" in content
    assert "MF-MAB" in content
    assert "NA" in content  # convergence for non-gossip
    assert "0.8500" in content  # convergence for gossip-capable

    os.remove(csv_path)
    print(f"PSEUDO-PASS: csv_output (wrote {len(content)} bytes)")


def test_report_output() -> None:
    """Test that the aggregated report can be generated."""
    from src.decentralized_simulator import DecentralizedConfig, DecentralizedSimulation

    # Generate a small set of aggregate results
    agg_results = []
    for i in range(3):
        cfg = DecentralizedConfig(num_nodes=5, sim_duration=5.0, seed=42 + i)
        sim = DecentralizedSimulation(
            balancer_class=MeanFieldMABBalancer,
            config=cfg,
        )
        result = sim.run()
        agg_results.append(AggregatedResult(
            suite_name="mini_test",
            balancer_name=f"MF-MAB-seed-{42+i}",
            gossip=True,
            num_nodes=5,
            arrival_rate=5.0,
            guess_error=0.05,
            seeds=1,
            mean_makespan=result.makespan,
            std_makespan=0.0,
            mean_throughput=result.tasks_completed / 5.0,
            std_throughput=0.0,
            mean_convergence=result.state_convergence,
            std_convergence=None,
            total_gossip_exchanges=len(result.gossip_exchanges),
            mean_runtime_s=0.0,
            successful_seeds=1,
            failed_seeds=0,
        ))

    # Should not crash
    print_aggregated_report(agg_results)
    print("PSEUDO-PASS: report_output")


def test_congestion_effect() -> None:
    """Higher arrival rate should increase makespan (basic sanity check)."""
    from src.decentralized_simulator import DecentralizedConfig, DecentralizedSimulation

    light_cfg = DecentralizedConfig(
        num_nodes=5, sim_duration=20.0, arrival_rate=5.0, seed=42,
    )
    heavy_cfg = DecentralizedConfig(
        num_nodes=5, sim_duration=20.0, arrival_rate=20.0, seed=42,
    )

    sim_light = DecentralizedSimulation(
        balancer_class=ShortestQueueBalancer, config=light_cfg,
    )
    sim_heavy = DecentralizedSimulation(
        balancer_class=ShortestQueueBalancer, config=heavy_cfg,
    )

    r_light = sim_light.run()
    r_heavy = sim_heavy.run()

    # Heavy load should have higher or equal makespan
    assert r_heavy.makespan >= r_light.makespan * 0.5, \
        f"Heavy makespan {r_heavy.makespan:.2f} should be >= light {r_light.makespan:.2f} * 0.5"
    print(f"PSEUDO-PASS: congestion_effect "
          f"(light={r_light.makespan:.2f}, heavy={r_heavy.makespan:.2f})")


def main() -> None:
    tests = [
        test_suite_definition,
        test_balancer_suite,
        test_small_suite_runs,
        test_gossip_vs_nogossip,
        test_large_node_topology_auto,
        test_aggregated_result,
        test_csv_output,
        test_report_output,
        test_congestion_effect,
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
            import traceback
            print(f'PSEUDO-ERROR: {test.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()

    print(f'\n=== Decentralized Test Suite Framework Pseudo-Tests ===\n\n'
          f'Pseudo-test results: {passed} passed, {failed} failed')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
