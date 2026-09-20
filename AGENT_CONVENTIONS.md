# Agent Conventions — Decentralized Load Balancer Simulation

## Main agent / subagent protocol
- I am the main agent. I spawn subagents to accomplish tasks (coding, testing, research).
- Confer with me about decisions/uncertainties promptly, but don't block on them — stack questions and continue unblocked work.
- Stay inside this folder and its subfolders (temp files OK, add to gitignore).
- Review subagent changes against past decisions; note important things as tests or "pseudotests."

## Project scope
- Decentralized load balancer simulation
- 4 core element categories besides the load balancing models:
  1. **Node descriptors** — identity, capacity, location, current load
  2. **Task descriptors** — type, size, deadline, dependencies
  3. **"True estimates" for workload time** — blackbox, stochastic to start
  4. **In-simulation "guessing" model** for workload time — what the load balancer uses (imperfect estimate)
- Find arXiv/papers on decentralized load balancers to implement, simulate, and test

## Episode cycle
Observe → Orient → Decide → Act → Verify → Loop

## Failure mode prevention
- Same tool fails 2+ times with same error class → STOP, switch strategy
- Verification succeeds 2+ times with identical output → say "done" and move on
- New input arrives → process it immediately, don't re-enter verification mode on prior work
- Subagent over-application: prompt must include what to change AND what stays unchanged
- Slow ops: verify source values BEFORE rebuilding/redeploying