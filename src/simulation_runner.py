"""Integrated simulation runner — wires engine + balancers + true model + metrics.

This is the top-level entry point for running a full discrete-event simulation:
task arrivals, balancer decisions, execution on nodes, completion callbacks,
and metrics collection — all in deterministic sim-time.

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
from src.oracle_balancer import OracleBalancer
from src.metrics import (
    makespan,
    avg_latency,
    tail_latency,
    jain_fairness,
    utilization,
    regret,
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
    regret: float
    oracle_makespan: float
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
            "regret": self.regret,
            "oracle_makespan": self.oracle_makespan,
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


def run_simulation(
    cfg: Config,
    balancer: Balancer,
    balancer_name: str = "unknown",
    oracle_balancer: OracleBalancer | None = None,
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
    pending_completions: dict[str, tuple[Task, Node, float]] = {}

    # --- Schedule task arrivals ---
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

    # --- Oracle pass (if oracle provided) ---
    oracle_makespan = 0.0
    if oracle_balancer is not None:
        oracle_nodes: dict[str, Node] = {}
        for i in range(cfg.num_nodes):
            node_id = f"node-{i}"
            oracle_nodes[node_id] = Node(
                id=node_id,
                cpu_cap=nodes[node_id].cpu_cap,
                memory_cap=nodes[node_id].memory_cap,
                network_cap=nodes[node_id].network_cap,
            )
        oracle_tasks = [
            Task(
                id=t.id,
                cpu_req=t.cpu_req,
                memory_req=t.memory_req,
                network_req=t.network_req,
                deadline=t.deadline,
                priority=t.priority,
                arrival_time=t.arrival_time,
            )
            for _, t in tasks
        ]
        oracle_model = StochasticTrueModel(
            seed=cfg.seed + 3000,
            noise_std=cfg.true_model_noise,
            cost_scale=cfg.true_model_cost_scale,
        )
        oracle_balancer._true_model = oracle_model
        for task in oracle_tasks:
            best_node = oracle_balancer.assign(task, list(oracle_nodes.values()))
            cost = oracle_model.cost(task, best_node)
            start_time = max(task.arrival_time, best_node.next_available_time)
            completion_time = start_time + cost
            best_node.next_available_time = completion_time
            best_node.assign(task.id)
            best_node.completed.append(task.id)
            best_node.total_busy_time += cost
        # Oracle makespan = max completion time across all nodes
        # (next_available_time tracks when each node finishes its last task)
        oracle_makespan = max(
            (n.next_available_time for n in oracle_nodes.values()), default=0.0
        )
        # FIX: oracle pass used next_available_time but it was already updated above.
        # The real issue: oracle_makespan was being overwritten to 0.0 in SimulationResult.
        # The oracle_makespan variable is correct — pass it through.
    node_list = list(nodes.values())
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
        regret=regret(node_list, task_arrivals, oracle_makespan),
        oracle_makespan=oracle_makespan,
        num_tasks=len(tasks),
        num_completions=len(completed_ids),
        sim_duration=cfg.sim_duration,
        event_log=sim.event_log,
    )
    return result


def compare_balancers(
    cfg: Config,
    balancers: list[tuple[str, Balancer]],
    oracle_balancer: OracleBalancer | None = None,
    num_seeds: int | None = None,
) -> list[SimulationResult]:
    """Run multiple balancers and collect results.

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
            result = run_simulation(
                run_cfg,
                balancer=balancer,
                balancer_name=f"{balancer_name}-s{seed_idx}",
                oracle_balancer=oracle_balancer,
            )
            all_results.append(result)
    return all_results


def print_results(results: list[SimulationResult]) -> None:
    """Print a formatted summary of simulation results."""
    print(f"\n{'='*72}")
    header = (
        f"{'Balancer':<30} {'Seed':<6} {'Makespan':>10} {'AvgLat':>8} "
        f"{'Tail99':>8} {'Jain':>6} {'Util':>6} {'Regret':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.balancer_name:<30} {r.seed:<6} {r.makespan:>10.3f} "
            f"{r.avg_latency:>8.3f} {r.tail_latency:>8.3f} "
            f"{r.jain_fairness:>6.3f} {r.utilization:>6.3f} {r.regret:>8.3f}"
        )
    print(f"{'='*72}\n")


if __name__ == "__main__":
    from src.random_balancer import RandomBalancer
    from src.round_robin_balancer import RoundRobinBalancer
    from src.shortest_queue_balancer import ShortestQueueBalancer
    from src.power_of_two_balancer import PowerOfTwoChoicesBalancer
    from src.work_stealing_balancer import WorkStealingBalancer

    cfg = Config(seed=42, num_nodes=5, sim_duration=50.0, num_seeds=1)

    oracle = OracleBalancer()
    balancers = [
        ("Random", RandomBalancer(seed=42)),
        ("RoundRobin", RoundRobinBalancer()),
        ("ShortestQueue", ShortestQueueBalancer()),
        ("PowerOfTwo", PowerOfTwoChoicesBalancer(seed=42)),
        ("WorkStealing", WorkStealingBalancer(seed=42)),
    ]

    results = compare_balancers(cfg, balancers, oracle_balancer=oracle)
    print_results(results)

    # Per-balancer averages
    print("Per-balancer averages:")
    for name, _ in balancers:
        name_results = [r for r in results if r.balancer_name.startswith(name)]
        if not name_results:
            continue
        avg_makespan = sum(r.makespan for r in name_results) / len(name_results)
        avg_regret = sum(r.regret for r in name_results) / len(name_results)
        avg_jain = sum(r.jain_fairness for r in name_results) / len(name_results)
        print(
            f"  {name:<20} makespan={avg_makespan:.3f}  "
            f"regret={avg_regret:.4f}  jain={avg_jain:.3f}"
        )
