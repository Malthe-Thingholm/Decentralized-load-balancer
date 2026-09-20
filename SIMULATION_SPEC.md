# Simulation Framework Spec

## Goal

**Decentralized load balancing research platform.** Simulate heterogeneous computing nodes, dispatch tasks, and compare balancing strategies against an oracle baseline and each other. Find and implement published decentralized load-balancing algorithms (from arXiv and other public sources), simulate them, and test their performance.

Originated Sunday, September 20, 2026. User preferences: TDD-style small changes with immediate verification; always rebuild + smoke + pixel-verify WASM/visual changes; confer early on decisions/uncertainties; work strictly inside this repo.

## Core Design

- Event-driven discrete sim-time (wall-clock independent)
- Black-box stochastic `TrueModel` (ground-truth cost, hidden from balancer)
- Per-node `GuessingModel` with optional gossip (imperfect estimate, what balancer uses)
- Configurable `Balancer` interface (strategy under test)
- Deterministic replay via seeds

## Entities

### Node
- Identity (`id`)
- Capability vector (`cpu_cap`, `memory_cap`, `network_cap`)
- Queue of assigned task IDs
- Completed task list
- `total_busy_time`, `current_load_end` for metrics
- Completion callback hook (`on_complete`)

### Task
- Identity (`id`)
- Resource requirements (`cpu_req`, `memory_req`, `network_req`)
- Optional `deadline`, `priority`, `arrival_time`
- Dependency tracking (future)

### TrueModel / TrueEstimator (interface + 2 implementations)
- `cost(task, node) -> float` — stochastic, called once per assignment
- Wrapped by assertion layer (`assert_cost`, `AssertionWrapper`) enforcing non-negative, finite, single-call-per-task
- Two implementations coexist:
  - `StochasticTrueModel` (Gaussian noise, resource-ratio base cost)
  - `TrueEstimator` (log-normal multiplicative noise)
- Seeded for deterministic replay

### GuessingModel (interface + 2 implementations)
- `estimate(task, node) -> float` — what the balancer actually sees
- `snapshot() -> dict` — for gossip exchange
- `merge(other_snapshot) -> None` — ingest peer state
- Two implementations:
  - `StochasticGuessingModel` (base formula + Gaussian error, gossip state)
  - `GuessingModel` (biased overestimate + heavier noise)
- Gap between true cost and guessed cost is the simulation's interesting dimension

### Balancer (interface)
- `assign(task, nodes) -> Node` — pick a node for a task
- `on_complete(task, node, actual_cost) -> None` — learn from completed work
- Implemented as a `Protocol` (structural subtyping)

## Event Engine

4 event types: `arrival`, `completion`, `gossip`, `balancer_tick`

Two engine implementations:
- `Simulator` (`src/event.py`) — dataclass `Event` with `sim_time`, `event_type`, `task`, `node_id`, `payload`; generator-based `run()`
- `Engine` (`sim/engine.py`) — enum `EventType`, dict-based handlers, `schedule(type, delay, payload)`

Both use priority queues ordered by sim-time with sequence tie-breaking. Deterministic replay via `random.Random(seed)`.

## Baselines

### Implemented
|| Balancer | File | Strategy |
|----------|------|----------|
|| Random | `src/random_balancer.py` | Uniform random assignment |
|| Round-robin | `src/round_robin_balancer.py` | Cycle through nodes in order |
|| Shortest-queue | `src/shortest_queue_balancer.py` | Pick node with fewest queued tasks |
|| Power-of-two-choices | `src/power_of_two_balancer.py` | Pick 2 random nodes, choose less loaded (k=2) |
|| Work-stealing (A2WS) | `src/work_stealing_balancer.py` | A2WS-inspired: idle nodes steal from overloaded; adaptive threshold |
|| MF-MAB bandit | `src/mf_mab_balancer.py` | UCB1 multi-armed bandit: explore/exploit task→node assignment |
|| REPS sprayer | `src/reps_balancer.py` | Adaptive probabilistic routing via entropy recycling |
|| RL Q-learning | `src/rl_balancer.py` | Tabular Q-learning: epsilon-greedy, online Q-updates |

### Baselines (offline reference)
|| Baseline | File | Strategy |
|----------|------|----------|
|| LPT batch (true) | `src/lpt_batch.py` | Graham's LPT: sort by descending cost, assign to earliest completion |
|| LPT batch (guess) | `src/lpt_batch.py` | Same but using imperfect guesses |
|| List scheduling (true) | `src/list_scheduling.py` | Online greedy: assign each task to earliest-completing node |
|| List scheduling (guess) | `src/list_scheduling.py` | Same but using imperfect guesses |
|| Lower bound | `src/lower_bounds.py` | max(work_bound, max_task_bound) — provable floor, no algorithm can beat it |

### Oracle (ground truth)
- `src/oracle_balancer.py` — `OracleBalancer`
- Single-task: pick cheapest node by true estimate
- Batch: brute-force enumeration for small instances (exponential; reference only)
- Used for regret comparison: `(actual_makespan - oracle_makespan) / oracle_makespan`

### To implement (from papers)
- A2WS work-stealing (arXiv 2401.04494)
- MF-MAB bandit balancer (arXiv 2407.00080)
- RL-based adaptive balancer (arXiv 2409.04896)
- Incentive-based balancer (arXiv 2501.01219) — optional, game-theoretic

## Metrics (`src/metrics.py`)

- **Makespan:** max `current_load_end` across nodes
- **Avg latency:** mean task completion time
- **Tail latency:** p99 (configurable percentile)
- **Jain's fairness:** over node utilizations (0–1)
- **Utilization:** total_busy_time / (num_nodes × sim_time)
- **Regret:** relative makespan gap vs oracle

## Simulation Configuration (`src/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | RNG seed |
| `num_nodes` | 5 | Number of compute nodes |
| `sim_duration` | 100.0 | Simulation duration |
| `balancer_tick_interval` | 10.0 | Balancer decision interval |
| `gossip_interval` | 20.0 | Gossip exchange interval |
| `task_arrival_rate` | 1.0 | Tasks per time unit |
| `task_cpu_mean` | 10.0 | Mean task CPU requirement |
| `task_mem_mean` | 10.0 | Mean task memory requirement |
| `task_net_mean` | 5.0 | Mean task network requirement |
| `node_cpu_range` | (20, 40) | Node CPU capacity range |
| `node_mem_range` | (20, 40) | Node memory capacity range |
| `node_net_range` | (10, 20) | Node network capacity range |
| `tail_latency_percentile` | 0.99 | Tail latency percentile |
| `num_seeds` | 3 | Multi-seed CI runs (Phase 2+) |
| `true_model_noise` | 0.1 | TrueModel noise std dev |
| `guess_model_error` | 0.2 | GuessingModel error std dev |

## Phased Rollout

### Phase 1: Core engine + TrueModel stub + assertion wrapper ✅ (done)
- `sim/` scaffold: event engine, TrueModel stub, assertion wrapper, smoke test
- `src/` parallel scaffold: event, task, node, balancer protocol, truesim, guess

### Phase 2: Baselines + metrics ✅ (done)
- Random, Round-robin, Shortest-queue, Power-of-two-choices ✅
- A2WS work-stealing ✅ (`src/work_stealing_balancer.py`)
- MF-MAB bandit ✅ (`src/mf_mab_balancer.py`)
- REPS sprayer ✅ (`src/reps_balancer.py`)
- RL Q-learning ✅ (`src/rl_balancer.py`)
- Offline baselines: LPT batch, list scheduling, lower bound ✅ (`src/lpt_batch.py`, `src/list_scheduling.py`, `src/lower_bounds.py`)
- Metrics ✅
- Smoke test ✅
- Runner with queueing model + lower-bound gap metric ✅ (`src/simulation_runner.py`)
- Multi-seed comparison ✅

### Phase 3: More papers + gossip
- Communication-efficient DML balancing (arXiv 2405.00839)
- Gossip mechanism for state sharing (all paper balancers already support `snapshot()`/`merge()`)
- QEdgeProxy multi-player bandits (arXiv 2512.18915) — optional
- Incentive-based game-theoretic LB (arXiv 2501.01219) — optional

### Phase 4: Sweep + analysis
- Confidence intervals on all metrics
- Regret/gap-to-LB comparison across all strategies
- Parameter sweeps (arrival rate, heterogeneity, noise levels)

## Papers to Implement

See `papers/README.md` and `papers/reads.md` for full notes. Summary:

| # | Paper | arXiv ID | Strategy | Target Phase |
|---|-------|----------|----------|--------------|
| 1 | Adaptive Asynchronous Work-Stealing (A2WS) | 2401.04494 | Work-stealing, smart victim selection, heterogeneous | 3 |
| 2 | Decentralized Task Offloading via Mean-Field MAB | 2407.00080 | Bandit-based task→node assignment, mean-field | 3 |
| 3 | Communication-Efficient DML Balancing | 2405.00839 | Heterogeneity-aware balancing for multi-agent learning | 3 |
| 4 | RL-Based Adaptive LB for Dynamic Cloud | 2409.04896 | RL agent learns assignment policy online | 3 |
| 5 | Open Decentralized Computational Network (Incentive-Based) | 2501.01219 | Game-theoretic, token/reputation incentives | 4 (optional) |
| 6 | REPS — Recycled Entropy Packet Spraying | 2407.21625 | Adaptive probabilistic routing via entropy | 3 |
| 7 | QEdgeProxy — Multi-Player Bandits for QoS | 2512.18915 | Multi-agent competitive bandit selection | 4 (optional) |

## Convention Notes

- **Working directory:** everything inside this repo and subfolders. Temp files go in `.gitignore`-d paths.
- **TDD-style:** small changes, immediate verification. Write a test or pseudo-test (noted description) for non-trivial additions.
- **Color header:** when asked to change colors, check the source-of-truth color header FIRST — downstream artifacts are often the only stale items.
- **WASM rebuild + browser smoke:** after any WASM/visual code change, always rebuild + smoke + pixel-verify before reporting done.
- **Decisions:** confer early/soon on uncertainties; otherwise continue unblocked work and stack questions for a decision session.
- **Loop prevention:** see `AGENTS.md/loop-prevention.md` and `AGENT_CONVENTIONS.md`.
