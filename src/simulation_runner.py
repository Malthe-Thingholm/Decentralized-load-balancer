"""Integrated simulation runner — wires engine + balancers + true model + metrics + baselines.

This is the top-level entry point for running a full discrete-event simulation:
task arrivals, balancer decisions, execution on nodes, completion callbacks,
and metrics collection — all in deterministic sim-time.

Baselines:
- ListSchedulingTrue: online greedy with true costs (oracle visibility, online)
- ListSchedulingGuess: online greedy with guessed costs (same info as real balancers)
- LPTTrue: offline LPT with true costs (best LPT can do with perfect info)
- LPTGuess: offline LPT with guessed costs (what LPT would see in practice)
- LowerBound: provable lower bound on optimal makespan (Graham 1969)

Usage:
    python -m src.simulation_runner
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.event import Event, Simulator, EVENT_ARRIVAL, EVENT_COMPLETION
from src.task import Task
from src.node import Node
from src.balancer import Balancer
from src.truesim import StochasticTrueModel
from src.guess import StochasticGuessingModel
from src.list_scheduling import ListSchedulingTrue, ListSchedulingGuess
from src.lpt_batch import LPTTrue, LPTGuess
from src.lower_bounds import combined_bound, relative_gap
from src.metrics import (
    makespan,
    avg_latency,
    tail_latency,
    jain_fairness,
    utilization,
)
from src.config import Config


@dataclass
class SimulationResult:
    """Collected metrics and trace from a single simulation run."""

    config: Config
    seed: int
    balancer_name: str
    makespan: float
    avg_latency: float
    tail_latency: float
    jain_fairness: float
    utilization: float
    lower_bound: float  # LPT lower bound on optimal makespan for this task set
    gap_to_lb: float  # (makespan - lb) / lb, always >= 0
    num_tasks: int
    num_completions: int
    sim_duration: float
    event_log: list[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "balancer": self.balancer_name,
            "seed": self.seed,
            "makespan": self.makespan,
            "avg_latency": self.avg_latency,
            "tail_latency": self.tail_latency,
            "jain_fairness": self.jain_fairness,
            "utilization": self.utilization,
            "lower_bound": self.lower_bound,
            "gap_to_lb": self.gap_to_lb,
            "num_tasks": self.num_tasks,
            "num_completions": self.num_completions,
            "sim_duration": self.sim_duration,
        }


def generate_tasks(
    cfg: Config,
    rng: random.Random,
    until: float,
) -> list[tuple[float, Task]]:
    """Generate task arrivals as (arrival_time, Task) pairs.

    Tasks arrive as a Poisson process with rate cfg.task_arrival_rate.
    Resource requirements are drawn from exponential distributions
    with means from config.
    """
    tasks: list[tuple[float, Task]] = []
    t = 0.0
    task_counter = 0
    while t < until:
        dt = rng.expovariate(cfg.task_arrival_rate)
        t += dt
        if t >= until:
            break
        task_counter += 1
        task = Task(
            id=f"task-{task_counter}",
            cpu_req=rng.expovariate(1.0 / cfg.task_cpu_mean),
            memory_req=rng.expovariate(1.0 / cfg.task_mem_mean),
            network_req=rng.expovariate(1.0 / cfg.task_net_mean),
            deadline=t + rng.uniform(20.0, 200.0),
            priority=rng.randint(0, 5),
            arrival_time=t,
        )
        tasks.append((t, task))
    return tasks


def compute_lower_bound(
    tasks: list[Task],
    nodes: list[Node],
    true_model: StochasticTrueModel,
) -> float:
    """Compute the LPT lower bound on optimal makespan for this task set.

    O(n*m) to compute. A valid lower bound — no algorithm can beat it.
    Uses true costs (the ground truth).
    """
    return combined_bound(tasks, nodes, true_model)


def run_simulation(
    cfg: Config,
    balancer: Balancer,
    balancer_name: str = "unknown",
    task_list: list[tuple[float, Task]] | None = None,
) -> SimulationResult:
    """Run a full discrete-event simulation with the given balancer.

    The Simulator is a priority-queue event loop with no handler registry.
    We drain its event generator and dispatch each event to the appropriate
    handler based on event_type.

    Parameters
    ----------
    cfg : Config
        Simulation configuration.
    balancer : Balancer
        The load balancing strategy to evaluate.
    balancer_name : str
        Human-readable name for the result.
    task_list : list[tuple[float, Task]] | None
        If provided, use this exact task sequence (from a seed-aligned
        generate_tasks call). Used by baseline runners to ensure all
        baselines see the same task set.

    Returns
    -------
    SimulationResult
        Collected metrics and trace.
    """
    rng = random.Random(cfg.seed)
    sim = Simulator(seed=cfg.seed)

    # --- Setup nodes ---
    nodes: dict[str, Node] = {}
    for i in range(cfg.num_nodes):
        node_id = f"node-{i}"
        node = Node(
            id=node_id,
            cpu_cap=rng.uniform(*cfg.node_cpu_range),
            memory_cap=rng.uniform(*cfg.node_mem_range),
            network_cap=rng.uniform(*cfg.node_net_range),
        )
        nodes[node_id] = node

    # --- TrueModel (ground truth cost) ---
    true_model = StochasticTrueModel(
        seed=cfg.seed + 1000,
        noise_std=cfg.true_model_noise,
        cost_scale=cfg.true_model_cost_scale,
    )

    # --- State tracking ---
    task_arrivals: dict[str, float] = {}
    completed_ids: set[str] = set()

    # --- Schedule task arrivals ---
    if task_list is not None:
        tasks = task_list
    else:
        tasks = generate_tasks(cfg, rng, cfg.sim_duration)

    for arrival_time, task in tasks:
        sim.schedule(
            Event(
                sim_time=arrival_time,
                event_type=EVENT_ARRIVAL,
                task=task,
            )
        )

    # --- Event loop ---
    for event in sim.run(until=cfg.sim_duration):
        if event.event_type == EVENT_ARRIVAL:
            task = event.task
            if task is None:
                continue
            task_arrivals[task.id] = sim.sim_time
            node_list = list(nodes.values())
            chosen = balancer.assign(task, node_list)
            chosen.record_arrival(task.id, sim.sim_time)
            chosen.assign(task.id)
            cost = true_model.cost(task, chosen)
            # Queueing: task starts when node is free (or now, whichever is later)
            start_time = max(sim.sim_time, chosen.next_available_time)
            completion_time = start_time + cost
            chosen.next_available_time = completion_time
            sim.schedule(
                Event(
                    sim_time=completion_time,
                    event_type=EVENT_COMPLETION,
                    task=task,
                    node_id=chosen.id,
                )
            )

        elif event.event_type == EVENT_COMPLETION:
            task = event.task
            node_id = event.node_id
            if task is None or node_id is None:
                continue
            node = nodes[node_id]
            actual_cost = true_model.cost(task, node)
            node.complete(task.id, sim.sim_time)
            balancer.on_complete(task, node, actual_cost)
            completed_ids.add(task.id)

    # --- Collect metrics ---
    node_list = list(nodes.values())
    lb = compute_lower_bound([t for _, t in tasks], node_list, true_model)
    result = SimulationResult(
        config=cfg,
        seed=cfg.seed,
        balancer_name=balancer_name,
        makespan=makespan(node_list),
        avg_latency=avg_latency(node_list, task_arrivals),
        tail_latency=tail_latency(
            node_list, task_arrivals, cfg.tail_latency_percentile
        ),
        jain_fairness=jain_fairness(node_list),
        utilization=utilization(node_list, cfg.sim_duration),
        lower_bound=lb,
        gap_to_lb=relative_gap(makespan(node_list), lb),
        num_tasks=len(tasks),
        num_completions=len(completed_ids),
        sim_duration=cfg.sim_duration,
        event_log=sim.event_log,
    )
    return result


def compare_balancers(
    cfg: Config,
    balancers: list[tuple[str, Balancer]],
    num_seeds: int | None = None,
) -> list[SimulationResult]:
    """Run multiple online balancers and collect results.

    Each balancer is run with the same task arrival sequence (same seed)
    for fair comparison. If num_seeds is provided, multiple seeds are used
    for confidence intervals.
    """
    if num_seeds is None:
        num_seeds = cfg.num_seeds

    all_results: list[SimulationResult] = []
    for balancer_name, balancer in balancers:
        for seed_idx in range(num_seeds):
            run_cfg = Config(
                seed=cfg.seed + seed_idx,
                num_nodes=cfg.num_nodes,
                sim_duration=cfg.sim_duration,
                balancer_tick_interval=cfg.balancer_tick_interval,
                gossip_interval=cfg.gossip_interval,
                task_arrival_rate=cfg.task_arrival_rate,
                task_cpu_mean=cfg.task_cpu_mean,
                task_mem_mean=cfg.task_mem_mean,
                task_net_mean=cfg.task_net_mean,
                node_cpu_range=cfg.node_cpu_range,
                node_mem_range=cfg.node_mem_range,
                node_net_range=cfg.node_net_range,
                tail_latency_percentile=cfg.tail_latency_percentile,
                num_seeds=cfg.num_seeds,
                true_model_noise=cfg.true_model_noise,
                guess_model_error=cfg.guess_model_error,
            )
            # Generate tasks once per seed so all baselines see the same set
            _rng = random.Random(run_cfg.seed)
            _tasks = generate_tasks(run_cfg, _rng, run_cfg.sim_duration)
            result = run_simulation(
                run_cfg,
                balancer=balancer,
                balancer_name=f"{balancer_name}-s{seed_idx}",
                task_list=_tasks,
            )
            all_results.append(result)
    return all_results


def run_baseline_comparison(
    cfg: Config,
    balancers: list[tuple[str, Balancer]],
    baselines: list[tuple[str, Balancer]],
    num_seeds: int | None = None,
) -> list[SimulationResult]:
    """Run offline baselines and collect results.

    Offline baselines (LPT) compute their schedule post-hoc on the same
    task set generated by this seed.
    """
    if num_seeds is None:
        num_seeds = cfg.num_seeds

    all_results: list[SimulationResult] = []

    for seed_idx in range(num_seeds):
        run_cfg = Config(
            seed=cfg.seed + seed_idx,
            num_nodes=cfg.num_nodes,
            sim_duration=cfg.sim_duration,
            balancer_tick_interval=cfg.balancer_tick_interval,
            gossip_interval=cfg.gossip_interval,
            task_arrival_rate=cfg.task_arrival_rate,
            task_cpu_mean=cfg.task_cpu_mean,
            task_mem_mean=cfg.task_mem_mean,
            task_net_mean=cfg.task_net_mean,
            node_cpu_range=cfg.node_cpu_range,
            node_mem_range=cfg.node_mem_range,
            node_net_range=cfg.node_net_range,
            tail_latency_percentile=cfg.tail_latency_percentile,
            num_seeds=cfg.num_seeds,
            true_model_noise=cfg.true_model_noise,
            guess_model_error=cfg.guess_model_error,
        )

        _rng = random.Random(run_cfg.seed)
        _nodes: dict[str, Node] = {}
        for i in range(run_cfg.num_nodes):
            node_id = f"node-{i}"
            _nodes[node_id] = Node(
                id=node_id,
                cpu_cap=_rng.uniform(*run_cfg.node_cpu_range),
                memory_cap=_rng.uniform(*run_cfg.node_mem_range),
                network_cap=_rng.uniform(*run_cfg.node_net_range),
            )

        _true_model = StochasticTrueModel(
            seed=run_cfg.seed + 1000,
            noise_std=run_cfg.true_model_noise,
            cost_scale=run_cfg.true_model_cost_scale,
        )

        _tasks = generate_tasks(run_cfg, _rng, run_cfg.sim_duration)
        _task_list = [t for _, t in _tasks]

        for baseline_name, baseline in baselines:
            makespan_val = baseline.makespan(_task_list, list(_nodes.values()))
            lb = compute_lower_bound(_task_list, list(_nodes.values()), _true_model)

            result = SimulationResult(
                config=run_cfg,
                seed=run_cfg.seed,
                balancer_name=f"{baseline_name}-s{seed_idx}",
                makespan=makespan_val,
                avg_latency=0.0,
                tail_latency=0.0,
                jain_fairness=0.0,
                utilization=0.0,
                lower_bound=lb,
                gap_to_lb=relative_gap(makespan_val, lb),
                num_tasks=len(_task_list),
                num_completions=len(_task_list),
                sim_duration=run_cfg.sim_duration,
            )
            all_results.append(result)

    return all_results


def print_results(results: list[SimulationResult]) -> None:
    """Print a formatted summary of simulation results."""
    print(f"\n{'='*90}")
    header = (
        f"{'Balancer':<30} {'Seed':<6} {'Makespan':>10} {'AvgLat':>8} "
        f"{'Tail99':>8} {'Jain':>6} {'Util':>6} {'LB':>8} {'Gap(LB)':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.balancer_name:<30} {r.seed:<6} {r.makespan:>10.3f} "
            f"{r.avg_latency:>8.3f} {r.tail_latency:>8.3f} "
            f"{r.jain_fairness:>6.3f} {r.utilization:>6.3f} "
            f"{r.lower_bound:>8.3f} {r.gap_to_lb:>8.3f}"
        )
    print(f"{'='*90}\n")

    # Group by balancer name, compute means
    from collections import defaultdict
    by_name: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in results:
        base = r.balancer_name.rsplit("-s", 1)[0]
        by_name[base].append(r)

    print("Per-balancer averages:")
    for name in sorted(by_name.keys()):
        results_list = by_name[name]
        avg_m = sum(r.makespan for r in results_list) / len(results_list)
        avg_g = sum(r.gap_to_lb for r in results_list) / len(results_list)
        avg_j = sum(r.jain_fairness for r in results_list) / len(results_list)
        print(
            f"  {name:<25} makespan={avg_m:.3f}  "
            f"gap_to_lb={avg_g:.4f}  jain={avg_j:.3f}"
        )


if __name__ == "__main__":
    from src.random_balancer import RandomBalancer
    from src.round_robin_balancer import RoundRobinBalancer
    from src.shortest_queue_balancer import ShortestQueueBalancer
    from src.power_of_two_balancer import PowerOfTwoChoicesBalancer
    from src.work_stealing_balancer import WorkStealingBalancer
    from src.list_scheduling import ListSchedulingTrue, ListSchedulingGuess
    from src.lpt_batch import LPTTrue, LPTGuess

    cfg = Config(seed=42, num_nodes=5, sim_duration=50.0, num_seeds=1)

    # Online balancers (run in event loop)
    online = [
        ("Random", RandomBalancer(seed=42)),
        ("RoundRobin", RoundRobinBalancer()),
        ("ShortestQueue", ShortestQueueBalancer()),
        ("PowerOfTwo", PowerOfTwoChoicesBalancer(seed=42)),
        ("WorkStealing", WorkStealingBalancer(seed=42)),
    ]

    # Seed models' RNGs so all balancers see identical cost estimates
    _true_model = StochasticTrueModel(
        seed=42 + 1000,
        noise_std=cfg.true_model_noise,
        cost_scale=cfg.true_model_cost_scale,
    )
    _guess_model = StochasticGuessingModel(
        seed=42 + 2000,
        error_std=cfg.guess_model_error,
    )
    _true_model.rng.seed(42 + 1000)
    _guess_model.rng.seed(42 + 2000)

    offline = [
        ("LPT-True", LPTTrue(_true_model)),
        ("LPT-Guess", LPTGuess(_guess_model)),
    ]

    results = compare_balancers(cfg, online)
    print_results(results)
