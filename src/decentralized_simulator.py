"""Decentralized load balancing simulation with gossip.

Each node runs its own instance of a load balancer (all same type).
Nodes gossip snapshots peer-to-peer. Tasks arrive at random nodes;
the receiving node's local balancer assigns the task to some node.

Key constraint: all nodes in a single simulation run the SAME balancer
type. Comparisons between balancer types happen via separate simulation runs.

Usage::

    from src.decentralized_simulator import (
        DecentralizedSimulation, DecentralizedConfig,
        compare_decentralized_balancers, DecentralizedResult,
    )
    from src.mf_mab_balancer import MF_MABBalancer

    cfg = DecentralizedConfig(num_nodes=5, sim_duration=50.0, seed=42)
    sim = DecentralizedSimulation(
        balancer_class=MF_MABBalancer, config=cfg,
    )
    result: DecentralizedResult = sim.run()
    print(result.makespan, result.gossip_exchanges)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.balancer import Balancer
from src.guess import StochasticGuessingModel
from src.node import Node
from src.task import Task
from src.truesim import StochasticTrueModel


# ---------------------------------------------------------------------------
# Decentralized node wrapper
# ---------------------------------------------------------------------------

@dataclass
class DecentralizedNode:
    """A node in the decentralized simulation.

    Wraps a compute ``Node`` with a locally-running ``Balancer`` instance.
    All nodes in a simulation share the same balancer *type* but have
    independent *instances* with their own learned state.
    """

    node_id: str
    node: Node
    balancer: Balancer
    gossip_neighbors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gossip protocol: balancers that can be gossiped
# ---------------------------------------------------------------------------

class GossipCapable(Protocol):
    """A balancer that supports snapshot/merge for gossip."""

    def snapshot(self) -> dict[str, Any]: ...
    def merge(self, other: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Gossip exchange record
# ---------------------------------------------------------------------------

@dataclass
class GossipExchange:
    """Result of one gossip round between two nodes."""

    sender_id: str
    receiver_id: str
    successful: bool = True


# ---------------------------------------------------------------------------
# Gossip functions
# ---------------------------------------------------------------------------

def _get_snapshot(bal: Balancer) -> dict[str, Any]:
    """Get a balancer's snapshot, handling both @property and method forms.

    Some balancers expose ``snapshot`` as a @property (e.g. MF-MAB),
    others as a method. This helper normalizes both.
    """
    snap = bal.snapshot
    if callable(snap):
        return snap()
    return snap  # type: ignore[return-value]


def gossip_pair(
    node_a: DecentralizedNode, node_b: DecentralizedNode
) -> GossipExchange | None:
    """Exchange gossip between two nodes (bidirectional merge).

    Returns None if either balancer doesn't support gossip.
    """
    ba, bb = node_a.balancer, node_b.balancer
    if not _is_gossip_capable(ba) or not _is_gossip_capable(bb):
        return None

    snap_a = _get_snapshot(ba)
    snap_b = _get_snapshot(bb)
    ba.merge(snap_b)
    bb.merge(snap_a)

    return GossipExchange(
        sender_id=node_a.node_id,
        receiver_id=node_b.node_id,
        successful=True,
    )


def gossip_broadcast(
    source: DecentralizedNode, peers: list[DecentralizedNode]
) -> list[GossipExchange]:
    """One node broadcasts its snapshot to all peers (peers merge source).

    Skips peers whose balancers don't support gossip.
    """
    if not _is_gossip_capable(source.balancer):
        return []

    snap = _get_snapshot(source.balancer)
    exchanges: list[GossipExchange] = []
    for peer in peers:
        if not _is_gossip_capable(peer.balancer):
            continue
        peer.balancer.merge(snap)
        exchanges.append(GossipExchange(
            sender_id=source.node_id,
            receiver_id=peer.node_id,
            successful=True,
        ))
    return exchanges


def _is_gossip_capable(bal: Balancer) -> bool:
    """Check if a balancer supports gossip (snapshot + merge).

    snapshot may be a @property (accessed, not called) or a method.
    merge must be a callable method.
    """
    has_snap = hasattr(bal, 'snapshot')
    has_merge = hasattr(bal, 'merge') and callable(bal.merge)
    if not (has_snap and has_merge):
        return False
    # snapshot: if it's a property, accessing it is enough;
    # if it's a method, it must be callable
    snap = bal.snapshot
    if callable(snap):
        return True
    # Not callable — it's a property, which is fine (accessed via .snapshot)
    return True


# ---------------------------------------------------------------------------
# Decentralized config
# ---------------------------------------------------------------------------

@dataclass
class DecentralizedConfig:
    """Configuration for decentralized simulation."""

    num_nodes: int = 5
    sim_duration: float = 50.0
    arrival_rate: float = 1.0
    true_model_cost_scale: float = 15.0
    true_model_noise: float = 0.1
    guess_model_error: float = 0.2
    seed: int = 42

    # Gossip
    gossip_interval: float = 10.0
    gossip_mode: str = 'pairwise_random'   # 'pairwise_random', 'broadcast_all', 'ring_pass'
    gossip_network: str = 'complete'       # 'complete', 'ring', 'random_k'

    # Node heterogeneity
    cpu_mean: float = 3.0
    cpu_std: float = 0.5
    memory_mean: float = 3.5
    memory_std: float = 0.4
    network_mean: float = 2.5
    network_std: float = 0.6


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def create_heterogeneous_nodes(
    num_nodes: int,
    rng: random.Random,
    cpu_mean: float = 3.0,
    cpu_std: float = 0.5,
    memory_mean: float = 3.5,
    memory_std: float = 0.4,
    network_mean: float = 2.5,
    network_std: float = 0.6,
) -> list[Node]:
    """Create heterogeneous compute nodes with log-normal capacity caps."""
    nodes = []
    for i in range(num_nodes):
        cpu_cap = rng.lognormvariate(cpu_mean, cpu_std)
        memory_cap = rng.lognormvariate(memory_mean, memory_std)
        network_cap = rng.lognormvariate(network_mean, network_std)
        node = Node(
            id=f"node-{i}",
            cpu_cap=cpu_cap,
            memory_cap=memory_cap,
            network_cap=network_cap,
        )
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Decentralized simulation runner
# ---------------------------------------------------------------------------

@dataclass
class DecentralizedResult:
    """Results from a decentralized simulation run."""

    makespan: float
    node_completions: dict[str, int]
    node_totals: dict[str, dict[str, float]]
    gossip_exchanges: list[GossipExchange]
    balancer_type: str
    config: DecentralizedConfig
    tasks_generated: int
    tasks_completed: int
    state_convergence: float | None = None
    avg_latency: float = 0.0
    tail_latency: float = 0.0
    node_utilization: dict[str, float] = field(default_factory=dict)
    task_latencies: list[float] = field(default_factory=list)


class DecentralizedSimulation:
    """Decentralized load balancing simulation with gossip.

    Each node runs its own balancer instance (all same type). Tasks arrive
    at random nodes; the local balancer assigns to some node. Nodes gossip
    periodically.

    All nodes in a single run use the SAME balancer type. To compare
    balancer types, run separate simulations.
    """

    def __init__(
        self,
        balancer_class: type[Balancer],
        balancer_kwargs: dict[str, Any] | None = None,
        config: DecentralizedConfig | None = None,
        true_model: StochasticTrueModel | None = None,
        guess_model: StochasticGuessingModel | None = None,
    ):
        self._balancer_class = balancer_class
        self._balancer_kwargs = balancer_kwargs or {}
        self._config = config or DecentralizedConfig()
        self._true_model = true_model
        self._guess_model = guess_model

        self._rng = random.Random(self._config.seed)
        self._task_rng = random.Random(self._config.seed + 1)

        # Runtime state
        self._nodes: list[DecentralizedNode] = []
        self._balancers: list[Balancer] = []
        self._next_available: dict[str, float] = {}
        self._current_time: float = 0.0
        self._gossip_exchanges: list[GossipExchange] = []
        self._tasks_generated: int = 0
        self._tasks_completed: int = 0
        self._node_completions: dict[str, int] = defaultdict(int)
        self._node_costs: dict[str, list[float]] = defaultdict(list)

        # Per-task latency and per-node busy time tracking
        self._task_latencies: list[float] = []
        self._node_busy_time: dict[str, float] = defaultdict(float)

        # Pending completions: list of (task_id, node_id, cost, completion_time, arrival_time)
        self._pending_completions: list[tuple[str, str, float, float, float]] = []

        if self._true_model is None:
            self._true_model = StochasticTrueModel(
                seed=self._config.seed + 10,
                cost_scale=self._config.true_model_cost_scale,
                noise_std=self._config.true_model_noise,
            )

        if self._guess_model is None:
            self._guess_model = StochasticGuessingModel(
                seed=self._config.seed + 20,
                error_std=self._config.guess_model_error,
            )

    def _create_nodes(self) -> None:
        """Create heterogeneous compute nodes and wrap with balancers."""
        raw_nodes = create_heterogeneous_nodes(
            self._config.num_nodes, self._rng,
            self._config.cpu_mean, self._config.cpu_std,
            self._config.memory_mean, self._config.memory_std,
            self._config.network_mean, self._config.network_std,
        )

        for node in raw_nodes:
            balancer = self._balancer_class(**self._balancer_kwargs)
            if hasattr(balancer, 'set_guess_model'):
                # type: ignore[attr-defined]
                balancer.set_guess_model(self._guess_model)

            dn = DecentralizedNode(
                node_id=node.id, node=node, balancer=balancer,
            )
            self._nodes.append(dn)
            self._balancers.append(balancer)
            self._next_available[node.id] = 0.0

        self._setup_gossip_network()

    def _setup_gossip_network(self) -> None:
        """Configure gossip neighbor relationships."""
        n = len(self._nodes)
        mode = self._config.gossip_network

        if mode == 'complete':
            for node in self._nodes:
                node.gossip_neighbors = [
                    o.node_id for o in self._nodes if o.node_id != node.node_id
                ]
        elif mode == 'ring':
            for i, node in enumerate(self._nodes):
                left = self._nodes[(i - 1) % n].node_id
                right = self._nodes[(i + 1) % n].node_id
                node.gossip_neighbors = [left, right]
        elif mode == 'random_k':
            import math
            k = max(1, int(math.sqrt(n)))
            for node in self._nodes:
                others = [o.node_id for o in self._nodes if o.node_id != node.node_id]
                node.gossip_neighbors = self._rng.sample(others, min(k, len(others)))
        else:
            raise ValueError(f"Unknown gossip_network mode: {mode}")

    def _generate_task(self) -> dict:
        """Generate a new task arriving at a random node."""
        arrival_node_id = self._rng.choice([n.node_id for n in self._nodes])
        cpu_req = max(0.1, self._task_rng.gauss(10.0, 3.0))
        memory_req = max(0.1, self._task_rng.gauss(10.0, 3.0))
        network_req = max(0.1, self._task_rng.gauss(5.0, 2.0))

        return {
            'id': f"task-{self._tasks_generated}",
            'cpu_req': cpu_req,
            'memory_req': memory_req,
            'network_req': network_req,
            'arrival_time': self._current_time,
            'arrival_node_id': arrival_node_id,
        }

    def _run_gossip(self) -> None:
        """Execute one gossip round based on gossip_mode."""
        mode = self._config.gossip_mode

        if mode == 'pairwise_random':
            available = list(self._nodes)
            self._rng.shuffle(available)
            for i in range(0, len(available) - 1, 2):
                ex = gossip_pair(available[i], available[i + 1])
                if ex is not None:
                    self._gossip_exchanges.append(ex)

        elif mode == 'broadcast_all':
            for node in self._nodes:
                peers = [self._get_node_by_id(nid) for nid in node.gossip_neighbors]
                self._gossip_exchanges.extend(gossip_broadcast(node, peers))

        elif mode == 'ring_pass':
            for i, node in enumerate(self._nodes):
                right_id = self._nodes[(i + 1) % len(self._nodes)].node_id
                ex = gossip_pair(node, self._get_node_by_id(right_id))
                if ex is not None:
                    self._gossip_exchanges.append(ex)

        else:
            raise ValueError(f"Unknown gossip_mode: {mode}")

    def _get_node_by_id(self, node_id: str) -> DecentralizedNode:
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        raise ValueError(f"Node {node_id} not found")

    def run(self) -> DecentralizedResult:
        """Run the decentralized simulation.

        Returns a DecentralizedResult with makespan, per-node stats,
        gossip exchange count, and state convergence metric.
        """
        self._create_nodes()
        self._current_time = 0.0
        self._tasks_generated = 0
        self._tasks_completed = 0
        self._node_completions = defaultdict(int)
        self._node_costs = defaultdict(list)
        self._pending_completions = []
        self._gossip_exchanges = []
        self._next_available = {n.node_id: 0.0 for n in self._nodes}

        next_arrival = self._rng.expovariate(self._config.arrival_rate)
        gossip_counter = 0

        while self._current_time < self._config.sim_duration:
            # Next gossip time
            gossip_time = (gossip_counter + 1) * self._config.gossip_interval
            if gossip_time > self._config.sim_duration:
                gossip_time = self._config.sim_duration + 1

            # Next event = min(arrival, gossip)
            next_event = min(next_arrival, gossip_time)
            if next_event > self._config.sim_duration:
                break

            self._current_time = next_event

            # Handle gossip
            if self._current_time >= gossip_time and gossip_time <= self._config.sim_duration:
                self._run_gossip()
                gossip_counter += 1

            # Handle task arrival
            if self._current_time >= next_arrival and next_arrival <= self._config.sim_duration:
                self._handle_task_arrival()
                next_arrival = self._current_time + self._rng.expovariate(
                    self._config.arrival_rate
                )

            # Process completions
            self._process_completions()

        makespan = max(self._next_available.values()) if self._next_available else 0.0

        # Build node totals
        node_totals = {}
        for nid in set(list(self._node_completions.keys()) + list(self._node_costs.keys())):
            costs = self._node_costs.get(nid, [])
            busy = self._node_busy_time.get(nid, 0.0)
            sim_dur = self._config.sim_duration
            node_totals[nid] = {
                'total_cost': sum(costs),
                'task_count': self._node_completions.get(nid, 0),
                'avg_cost': sum(costs) / len(costs) if costs else 0.0,
                'utilization': busy / sim_dur if sim_dur > 0 else 0.0,
            }

        # Per-task latency stats
        all_latencies = list(self._task_latencies)
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
        tail_latency = (
            sorted(all_latencies)[int(len(all_latencies) * 0.99)]
            if len(all_latencies) >= 100 else
            (sorted(all_latencies)[-1] if all_latencies else 0.0)
        )

        convergence = self._measure_state_convergence()

        return DecentralizedResult(
            makespan=makespan,
            node_completions=dict(self._node_completions),
            node_totals=node_totals,
            gossip_exchanges=self._gossip_exchanges,
            balancer_type=self._balancer_class.__name__,
            config=self._config,
            tasks_generated=self._tasks_generated,
            tasks_completed=self._tasks_completed,
            state_convergence=convergence,
            avg_latency=avg_latency,
            tail_latency=tail_latency,
            node_utilization=dict(
                (nid, data['utilization']) for nid, data in node_totals.items()
            ),
            task_latencies=all_latencies,
        )

    def _handle_task_arrival(self) -> None:
        """Handle a single task arrival event."""
        task = self._generate_task()
        self._tasks_generated += 1

        # The node that received the task decides where to send it
        recv = self._get_node_by_id(task['arrival_node_id'])
        task_obj = Task(
            id=task['id'],
            cpu_req=task['cpu_req'],
            memory_req=task['memory_req'],
            network_req=task['network_req'],
            deadline=None,
            priority=0,
        )

        chosen = recv.balancer.assign(task_obj, [n.node for n in self._nodes])
        target_id = chosen.id
        target_node = self._get_node_by_id(target_id).node

        # Update target node's queue_depth (so balancers see accurate state)
        target_node.assign(task['id'])

        # Compute true cost
        cost = self._true_model.cost(task_obj, target_node)

        # Schedule completion
        start_time = max(self._current_time, self._next_available[target_id])
        completion_time = start_time + cost
        self._next_available[target_id] = completion_time

        self._pending_completions.append(
            (task['id'], target_id, cost, completion_time, task['arrival_time'])
        )

    def _process_completions(self) -> None:
        """Process all task completions whose time has come."""
        still_pending: list[tuple[str, str, float, float, float]] = []

        for task_id, node_id, cost, completion_time, arrival_time in self._pending_completions:
            if completion_time <= self._current_time:
                self._tasks_completed += 1
                self._node_completions[node_id] += 1
                self._node_costs[node_id].append(cost)

                # Per-task latency tracking
                latency = completion_time - arrival_time
                self._task_latencies.append(latency)

                # Per-node busy time tracking
                start_time = max(arrival_time, self._next_available.get(node_id, 0.0))
                busy_time = completion_time - start_time
                self._node_busy_time[node_id] += max(0.0, busy_time)

                target_node = self._get_node_by_id(node_id).node
                task_obj = Task(
                    id=task_id,
                    cpu_req=0, memory_req=0, network_req=0,
                    deadline=None, priority=0,
                )
                # Notify all gossip-capable balancers
                for dn in self._nodes:
                    if _is_gossip_capable(dn.balancer):
                        dn.balancer.on_complete(task_obj, target_node, cost)
            else:
                still_pending.append((task_id, node_id, cost, completion_time, arrival_time))

        self._pending_completions = still_pending

    def _measure_state_convergence(self) -> float | None:
        """Measure how similar nodes' balancer states are after gossip.

        Returns a value in [0, 1] where 1 = perfectly converged (all nodes
        have identical assignment counts). Only works for gossip-capable
        balancers that have actually exchanged state.

        Tries 'total_assignments' first (most balancers), then 'total_pulls'
        (MF-MAB), then falls back to the largest numeric key in the snapshot.
        """
        # Only measure convergence if gossip actually happened
        if not self._gossip_exchanges:
            return None

        gossip_balancers = [b for b in self._balancers if _is_gossip_capable(b)]
        if len(gossip_balancers) < 2:
            return None

        snaps = [_get_snapshot(b) for b in gossip_balancers]

        # Find the key to use: prefer total_assignments, then total_pulls,
        # then any key with 'assign' or 'pull' in the name
        def _get_count(snap: dict) -> int:
            for key in ('total_assignments', 'total_pulls'):
                if key in snap:
                    return int(snap[key])
            # Fallback: find any key whose value is an int > 0
            for k, v in snap.items():
                if isinstance(v, (int, float)) and v > 0 and 'assign' in k.lower():
                    return int(v)
            return 0

        totals = [_get_count(s) for s in snaps]
        if not totals or all(t == 0 for t in totals):
            return None

        mean = sum(totals) / len(totals)
        var = sum((t - mean) ** 2 for t in totals) / len(totals)
        std = math.sqrt(var)
        if mean == 0:
            return None

        cv = std / mean  # coefficient of variation
        return max(0.0, 1.0 - cv)


# ---------------------------------------------------------------------------
# Multi-run comparison (for comparing balancer types)
# ---------------------------------------------------------------------------

def compare_decentralized_balancers(
    balancer_classes: list[type[Balancer]],
    balancer_kwargs_list: list[dict[str, Any]] | None = None,
    config: DecentralizedConfig | None = None,
    num_seeds: int = 3,
) -> list[DecentralizedResult]:
    """Run decentralized simulations for multiple balancer types and seeds.

    All nodes within a single run use the SAME balancer type.
    Different balancer types are compared across separate runs.

    Args:
        balancer_classes: List of balancer classes to compare.
        balancer_kwargs_list: Optional per-balancer kwargs list.
        config: Shared configuration.
        num_seeds: Number of random seeds per balancer type.

    Returns:
        List of DecentralizedResult for each (balancer, seed) combination.
    """
    if balancer_kwargs_list is None:
        balancer_kwargs_list = [{} for _ in balancer_classes]

    if len(balancer_kwargs_list) != len(balancer_classes):
        raise ValueError(
            "balancer_kwargs_list must match balancer_classes length"
        )

    results: list[DecentralizedResult] = []
    base_config = config or DecentralizedConfig()

    for balancer_cls, kwargs in zip(balancer_classes, balancer_kwargs_list):
        for seed_idx in range(num_seeds):
            cfg_kwargs = {
                k: v for k, v in base_config.__dict__.items()
                if k != 'seed'
            }
            cfg = DecentralizedConfig(
                **cfg_kwargs,
                seed=base_config.seed + seed_idx * 1000,
            )
            sim = DecentralizedSimulation(
                balancer_class=balancer_cls,
                balancer_kwargs=kwargs,
                config=cfg,
            )
            results.append(sim.run())

    return results


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from src.random_balancer import RandomBalancer
    from src.shortest_queue_balancer import ShortestQueueBalancer
    from src.mf_mab_balancer import MeanFieldMABBalancer
    from src.dml_balancer import CommunicationEfficientDMLBalancer
    from src.qedgeproxy_balancer import QEdgeProxyBalancer
    from src.incentive_balancer import IncentiveBalancer

    print('=== Decentralized Simulation Demo ===\n')

    cfg = DecentralizedConfig(
        num_nodes=5,
        sim_duration=50.0,
        arrival_rate=1.0,
        gossip_interval=10.0,
        gossip_mode='pairwise_random',
        gossip_network='complete',
        seed=42,
    )

    # Single run with RandomBalancer
    sim = DecentralizedSimulation(balancer_class=RandomBalancer, config=cfg)
    result = sim.run()

    print(f"Balancer: {result.balancer_type}")
    print(f"Makespan: {result.makespan:.2f}")
    print(f"Tasks: {result.tasks_completed}/{result.tasks_generated}")
    print(f"Gossip exchanges: {len(result.gossip_exchanges)}")
    print(f"State convergence: {result.state_convergence}")
    print(f"Node completions: {result.node_completions}")
    print(f"Node avg costs: ", end='')
    for nid in sorted(result.node_totals):
        t = result.node_totals[nid]
        print(f"{nid}={t.get('avg_cost', 0):.2f}", end=' ')
    print()

    # MF-MAB with gossip
    print('\n=== MF-MAB with gossip ===\n')
    sim2 = DecentralizedSimulation(balancer_class=MeanFieldMABBalancer, config=cfg)
    r2 = sim2.run()
    print(f"Balancer: {r2.balancer_type}")
    print(f"Makespan: {r2.makespan:.2f}")
    print(f"Tasks: {r2.tasks_completed}/{r2.tasks_generated}")
    print(f"Gossip exchanges: {len(r2.gossip_exchanges)}")
    print(f"State convergence: {r2.state_convergence}")

    # Compare
    print('\n=== Comparing Random vs ShortestQueue (2 seeds each) ===\n')
    results = compare_decentralized_balancers(
        [RandomBalancer, ShortestQueueBalancer],
        config=cfg,
        num_seeds=2,
    )
    for r in results:
        print(f"{r.balancer_type} (seed={r.config.seed}): "
              f"makespan={r.makespan:.2f}, "
              f"completed={r.tasks_completed}/{r.tasks_generated}, "
              f"convergence={r.state_convergence}")
