# Decentralized Load Balancer — Simulation Framework

## Project structure (TBD — scaffold below)
- `src/node_descriptors.py` — node identity, capacity, location, current load
- `src/task_descriptors.py` — task type, size, deadline, dependencies
- `src/true_estimates.py` — blackbox stochastic workload time estimator
- `src/guessing_model.py` — in-simulation imperfect estimate used by load balancer
- `src/balancer.py` — load balancing logic (to be implemented per paper)
- `src/simulation.py` — main event loop tying it all together
- `tests/` — unit tests + integration tests
- `papers/` — arxiv paper summaries + implementation notes

## Core element categories
1. **Node descriptors** — identity, capacity, location, current load
2. **Task descriptors** — type, size, deadline, dependencies
3. **True estimates** — blackbox, stochastic workload time (ground truth)
4. **Guessing model** — in-simulation imperfect estimate (what balancer uses)

## Papers to implement
- BON (cs/0411046) — balanced overlay networks
- Decentralized Task Offloading (2407.00080)
- Incentive-Based LB (2501.01219)
- Comm-Efficient DML Balancing (2405.00839)
- RL-Based Adaptive LB (2409.04896)