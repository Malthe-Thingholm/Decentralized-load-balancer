# Simulation Framework Spec

## Goal

Decentralized load balancing research platform: simulate heterogeneous computing nodes, dispatch tasks, compare balancing strategies against oracle baseline and each other.

## Core Design

- Event-driven discrete sim-time (wall-clock independent)
- Black-box stochastic TrueModel
- Per-node GuessingModel with optional gossip
- Configurable Balancer interface

## Entities

### Node
- Capability vector (CPU, memory, network)
- Queue of assigned tasks
- Local GuessingModel state
- Completion callback hook

### Task
- Requirements vector
- Optional deadline/priority
- Arrives via event queue

### TrueModel (interface)
- `cost(task, node) -> float` — stochastic, called once per assignment, wrapped by assertion

### GuessingModel (interface)
- `estimate(task, node) -> float`
- `snapshot() -> dict` — for gossip
- `merge(other_snapshot) -> None`

### Balancer (interface)
- `assign(task, nodes) -> node`
- `on_complete(task, node, actual_cost) -> None`

## Event Engine

4 event types: `arrival`, `completion`, `gossip`, `balancer_tick`
Priority queue ordered by sim-time. Deterministic replay via seeds.

## Baselines

Oracle (Hungarian), Random, Round-robin, Power-of-two-choices, Shortest-queue — extensible.

## Metrics

Makespan, avg/tail latency, Jain's fairness, utilization, regret (vs oracle). Time series + distributions, multi-seed CIs.

## Phased Rollout

1. Core engine + TrueModel stub + assertion wrapper
2. Baselines + metrics
3. Gossip + DRL balancers
4. Sweep harness
