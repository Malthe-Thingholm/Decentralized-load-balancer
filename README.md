# Decentralized Load Balancer — Simulation Project

## Purpose

Test and compare **decentralized load balancing strategies** via discrete-event simulation. The simulator models a population of heterogeneous compute nodes, a stream of arriving tasks, and competing assignment policies — all evaluated against an oracle (optimal) baseline and each other.

## How to Use This Repository

- **Run the smoke test:** `python -m src.smoke_test` (or `python src/smoke_test.py`)
- **Run unit tests:** `python -m pytest tests/ -v`
- **Read the spec:** `SIMULATION_SPEC.md`
- **Read papers:** `papers/README.md` and `papers/reads.md`
- **Experiment:** tweak `src/config.py`, drop a new balancer in `src/`, and rerun

## Project Structure

```
.
├── src/
│   ├── config.py              # Simulation configuration (seed, node/task params, intervals)
│   ├── event.py               # Event dataclass + Simulator (priority-queue event loop)
│   ├── task.py                # Task dataclass (resource requirements, deadline, priority)
│   ├── node.py                # Node dataclass (capability vector, queue, completion hook)
│   ├── balancer.py            # Balancer protocol (interface all strategies implement)
│   ├── guess.py               # GuessingModel — imperfect in-simulation estimator
│   ├── truesim.py             # TrueModel — stochastic black-box ground-truth cost
│   ├── true_estimates.py      # TrueEstimator — alternative black-box (log-normal noise)
│   ├── guessing_model.py      # GuessingModel — alternative imperfect estimator (bias+noise)
│   ├── node_descriptors.py    # Node descriptors (legacy/alternative node model)
│   ├── task_descriptors.py    # Task descriptors (legacy/alternative task model)
│   ├── metrics.py             # Makespan, latency, Jain's fairness, utilization, regret
│   ├── oracle_balancer.py     # Oracle (Hungarian/optimal) baseline
│   ├── random_balancer.py     # Random assignment baseline
│   ├── round_robin_balancer.py # Round-robin baseline
│   ├── shortest_queue_balancer.py # Shortest-queue baseline
│   ├── power_of_two_balancer.py # Power-of-two-choices baseline
│   ├── work_stealing_balancer.py # A2WS-inspired work-stealing balancer
│   ├── __init__.py
│   └── smoke_test.py          # Deterministic smoke test (no network/external deps)
├── sim/                        # Phase 1 scaffold (alternate event-engine design)
│   ├── __init__.py
│   ├── engine.py               # Event engine (enum-based events, handlers)
│   ├── true_model.py           # TrueModel stub (dict-based Task/Node)
│   ├── assertions.py           # AssertionWrapper for TrueModel invariants
│   ├── guessing_model.py       # GuessingModel stub (gossip snapshot/merge)
│   ├── balancer.py             # Balancer interface + Random/RoundRobin/ShortestQueue
│   ├── types.py                # Shared types (Capability, Task, NodeState, Event)
│   └── smoke_test.py           # Phase 1 smoke test
├── tests/
│   └── test_phase1.py          # Phase 1 tests (engine + TrueModel + assertions)
├── papers/
│   ├── README.md               # Paper collection overview
│   └── reads.md                # Detailed paper notes + implementation relevance
├── AGENTS.md/                  # Agent conventions (loop prevention, verification)
├── AGENT_CONVENTIONS.md        # Project-level agent conventions
├── SIMULATION_SPEC.md          # Full framework spec (entities, events, phased rollout)
├── LICENSE
└── .gitignore
```

## Core Element Categories

Every simulation run combines these five elements:

### 1. Load Balancing Models (`src/*.py` — balancers)
The strategy under test. Each implements the `Balancer` protocol:
```python
class Balancer(Protocol):
    def assign(self, task: Task, nodes: list[Node]) -> Node: ...
    def on_complete(self, task: Task, node: Node, actual_cost: float) -> None: ...
```
Baselines included: Random, Round-robin, Shortest-queue, Power-of-two-choices. Oracle (optimal) provided separately for regret comparison.

### 2. Node Descriptors (`src/node.py`)
Each node has:
- **Identity:** `id`
- **Capability vector:** `cpu_cap`, `memory_cap`, `network_cap`
- **Queue:** pending task IDs
- **Load tracking:** `current_load_end`, `total_busy_time`
- **Completion hook:** `on_complete` callback

### 3. Task Descriptors (`src/task.py`)
Each task has:
- **Identity:** `id`
- **Resource requirements:** `cpu_req`, `memory_req`, `network_req`
- **Deadline:** `deadline` (sim-time)
- **Priority:** `priority`
- **Arrival time:** `arrival_time`

### 4. True Estimates — Black-Box Stochastic Workload Time (`src/truesim.py`, `src/true_estimates.py`)
The **hidden oracle** that determines actual execution cost. The load balancer never sees this — it only gets imperfect guesses. Two implementations coexist:
- `StochasticTrueModel`: Gaussian noise around a resource-ratio base cost; wrapped by `assert_cost` to enforce non-negativity.
- `TrueEstimator`: Log-normal multiplicative noise; separate codebase for comparison.

Both are stochastic and seeded for deterministic replay.

### 5. In-Simulation Guessing Model (`src/guess.py`, `src/guessing_model.py`)
What the load balancer actually uses to make decisions. Two implementations:
- `StochasticGuessingModel`: same base formula as TrueModel but with estimation error (Gaussian noise); supports gossip `snapshot()`/`merge()`.
- `GuessingModel`: biased overestimate + heavier Gaussian noise; separate codebase.

The gap between true cost and guessed cost is what makes the simulation interesting — a perfect estimator reduces every balancer to the oracle.

## Architecture

**Event-driven discrete simulation:**
- 4 event types: `arrival`, `completion`, `gossip`, `balancer_tick`
- Priority queue ordered by sim-time (wall-clock independent)
- Deterministic replay via seeds (`random.Random(seed)` everywhere)
- `Simulator` (`src/event.py`) and `Engine` (`sim/engine.py`) are two implementations of the same idea

**Phased rollout:**
1. Core engine + TrueModel stub + assertion wrapper ✅ (done in both `sim/` and `src/`)
2. Baselines + metrics ✅ (baselines in `src/`, metrics in `src/metrics.py`)
3. Gossip + DRL balancers (future)
4. Sweep harness for multi-seed comparison (future)

## Papers to Implement

See `papers/reads.md` for detailed notes. The 5 candidate papers:

| # | Paper | arXiv ID | Strategy | Phase |
|---|-------|----------|----------|-------|
| 1 | Adaptive Asynchronous Work-Stealing (A2WS) | 2401.04494 | Work-stealing with smart victim selection | 2 |
| 2 | Decentralized Task Offloading via Mean-Field MAB | 2407.00080 | Bandit-based task→node assignment | 2–3 |
| 3 | Communication-Efficient DML Balancing | 2405.00839 | Heterogeneity-aware balancing for multi-agent learning | 2 |
| 4 | RL-Based Adaptive LB for Dynamic Cloud | 2409.04896 | RL-based adaptive load balancing | 3 |
| 5 | Open Decentralized Computational Network (Incentive-Based) | 2501.01219 | Game-theoretic/incentive-driven LB | 3–4 |

## Multi-Seed Comparison

Config supports `num_seeds` (default 3). Future sweep harness will run each balancer across multiple seeds and report means + confidence intervals for all metrics.

## Conventions

- **Working directory:** everything happens inside this repo and subfolders. Temporary files go in `.gitignore`-d paths.
- **TDD-style:** small changes, immediate verification. Write a test or pseudo-test (noted description) for non-trivial additions.
- **Color header:** if asked to change colors, check the source-of-truth color header FIRST — downstream artifacts (tests, docs, CSS) are often the only stale items.
- **WASM rebuild + browser smoke:** after code changes to any WASM/visual component, always rebuild + smoke + pixel-verify before reporting done.
- **Decisions:** confer with the user immediately/shortly after prompting when there's uncertainty; otherwise continue unblocked work and stack questions for a decision session.
- **Loop prevention:** see `AGENTS.md/loop-prevention.md` and `AGENT_CONVENTIONS.md`.

## License

MIT — see `LICENSE`.
