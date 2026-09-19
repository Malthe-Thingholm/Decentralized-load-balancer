"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Phase 1 simulation configuration."""

    seed: int = 42
    num_nodes: int = 5
    sim_duration: float = 100.0
    balancer_tick_interval: float = 10.0
    gossip_interval: float = 20.0
    task_arrival_rate: float = 1.0  # tasks per time unit
    task_cpu_mean: float = 10.0
    task_mem_mean: float = 10.0
    task_net_mean: float = 5.0
    node_cpu_range: tuple[float, float] = (20.0, 40.0)
    node_mem_range: tuple[float, float] = (20.0, 40.0)
    node_net_range: tuple[float, float] = (10.0, 20.0)

    # Metrics
    tail_latency_percentile: float = 0.99
    num_seeds: int = 3  # for multi-seed CIs (Phase 2+)

    # TrueModel
    true_model_noise: float = 0.1  # std dev of stochastic cost noise

    # GuessingModel
    guess_model_error: float = 0.2  # base estimation error std dev