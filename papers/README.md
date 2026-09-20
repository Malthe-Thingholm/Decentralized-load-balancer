# Decentralized Load Balancer Papers

## Collected from Arxiv (Phase 1 research)

### 1. Adaptive Asynchronous Work-Stealing (A2WS)
- **URL:** https://arxiv.org/abs/2401.04494
- **Title:** Adaptive Asynchronous Work-Stealing for distributed load-balancing in heterogeneous systems
- **Date:** January 2024 (v2, updated Jan 23)
- **Domain:** cs.DC (Distributed, Parallel, and Cluster Computing)
- **Authors:** João B. Fernandes, Ítalo A. S. de Assis, Idalmis M. S. Martins, Tiago Barros, Samuel Xavier-de-Souza
- **Key idea:** Smart victim selection + limited info communication + asynchronous theft. Adapts to heterogeneity without central coordinator.
- **Relevance to sim:** Candidate for work-stealing baseline and gossip-based state estimation. Maps to: decentralized balancer that lets idle nodes "steal" from busy ones using partial state.
- **Implementation notes:** Need to model: victim selection heuristic, steal request/reply protocol, adaptation to node capability differences.

### 2. Recycled Entropy Packet Spraying (REPS)
- **URL:** https://arxiv.org/abs/2407.21625
- **Title:** REPS: Adaptive Load Balancing Through Recycled Entropy Packet Spraying
- **Date:** July 2024
- **Domain:** cs.NI (Networking and Internet Architecture)
- **Key idea:** Entropy-based packet spraying that recycles load information to maintain adaptivity. Stochastic routing guided by load signals.
- **Relevance to sim:** Candidate for adaptive probabilistic routing baseline; tests whether entropy-based routing beats simple random.
- **Implementation notes:** Models probabilistic task routing where probabilities are adjusted based on observed load feedback.

### 3. Mean-Field Multi-Armed Bandits (MF-MAB)
- **URL:** https://arxiv.org/abs/2407.00080
- **Title:** Decentralized Task Offloading and Load-Balancing for Mobile Edge Computing in Dense Networks
- **Date:** June 2024
- **Domain:** cs.DC, cs.LG, cs.MA
- **Authors:** Mariam Yahya, Alexander Conzelmann, Setareh Maghsudi
- **Key idea:** Each IoT device acts as a bandit deciding which edge server to offload to; mean-field approximation handles the population size. Solves unknown network info + random task sizes.
- **Relevance to sim:** Candidate for bandit-based load balancer; maps directly to task→node assignment with exploration/exploitation tradeoff.
- **Implementation notes:** Each node runs a bandit algorithm; mean-field term captures aggregate population behavior. Exploration vs exploitation tradeoff controlled by confidence bounds.

### 4. QoS-Aware Load Balancing via Multi-Player Bandits (QEdgeProxy)
- **URL:** https://arxiv.org/abs/2512.18915
- **Title:** QEdgeProxy: QoS-Aware Load Balancing in Edge Computing via Multi-Player Multi-Armed Bandits
- **Date:** December 2025
- **Domain:** cs.DC, cs.NI
- **Key idea:** Multiple proxy nodes independently select edge servers using bandits; handles collisions when multiple proxies pick the same server.
- **Relevance to sim:** Candidate for multi-agent competitive assignment; tests collision-avoidance in decentralized selection.
- **Implementation notes:** Multi-player bandit with collision handling. Each proxy independently selects; collision resolution needed when two proxies pick same server.

### 5. Incentive-Based Decentralized LB
- **URL:** https://arxiv.org/abs/2501.01219
- **Title:** Model of an Open, Decentralized Computational Network with Incentive-Based Load Balancing
- **Date:** January 2025
- **Domain:** q-fin.CP, math.DS, math.OC
- **Authors:** German Rodikov
- **Key idea:** Load balancing driven by economic incentives (tokens/pricing) rather than pure algorithmic heuristics. Permissionless, decentralized network for complex computations.
- **Relevance to sim:** Candidate for game-theoretic baseline; useful if we want to test incentive layers later.
- **Implementation notes:** Requires modeling token/reputation mechanics. Tasks have costs; nodes have incentives to accept or reject. Equilibrium analysis needed.

---

## Additional Papers from Project Roadmap

### 6. Communication-Efficient DML Balancing
- **URL:** https://arxiv.org/abs/2405.00839
- **Title:** Communication-Efficient Training Workload Balancing for Decentralized Multi-Agent Learning
- **Date:** May 2024
- **Domain:** cs.LG, cs.AI, cs.DC, cs.MA, cs.PF
- **Authors:** Seyed Mahmoud Sajjadi Mohammadabadi, Lei Yang, Feng Yan, Junshan Zhang
- **Key idea:** Addresses heterogeneity in agents' resources (computation, communication, task size) that creates bottlenecks in decentralized multi-agent learning. Balances training workload across agents.
- **Relevance to sim:** Heterogeneity-aware balancing; maps to nodes with different capability vectors. Tasks have different sizes; assigns to minimize straggler effect.
- **Implementation notes:** Focus on balancing workload given heterogeneous node capacities and variable task sizes. May involve grouping/partitioning strategies.

### 7. RL-Based Adaptive LB for Dynamic Cloud
- **URL:** https://arxiv.org/abs/2409.04896
- **Title:** Reinforcement Learning-Based Adaptive Load Balancing for Dynamic Cloud Environments
- **Date:** September 2024
- **Domain:** cs.DC, cs.AI, cs.NI
- **Authors:** Kavish Chawla
- **Key idea:** RL agent learns to balance load dynamically in cloud environments where traditional algorithms (round-robin, least-connections) are static and can't adapt to changing conditions.
- **Relevance to sim:** RL-based balancer that learns from observed costs. State = node loads + queue depths; action = which node to assign to; reward = negative latency/cost.
- **Implementation notes:** RL agent state space, action space, reward function design. Online learning during simulation vs offline training.

---

## Next Steps
1. ✅ Baselines implemented: Random, Round-robin, Shortest-queue, Power-of-two-choices
2. ✅ A2WS work-stealing implemented (`src/work_stealing_balancer.py`) — first adaptive baseline
3. ✅ MF-MAB bandit implemented (`src/mf_mab_balancer.py`) — UCB1 exploration/exploitation
4. ✅ REPS sprayer implemented (`src/reps_balancer.py`) — entropy-based adaptive routing
5. ✅ RL Q-learning implemented (`src/rl_balancer.py`) — tabular Q-learning online
6. ✅ Lower bound + LPT batch + list scheduling (`src/lower_bounds.py`, `src/lpt_batch.py`, `src/list_scheduling.py`)
7. ✅ Sweep-ready runner (`src/simulation_runner.py`) — multi-seed, lower-bound gap metric
8. Implement communication-efficient DML balancing (2405.00839)
9. Gossip mechanism for decentralized state sharing (all balancers support snapshot/merge)

---

## Paper Search Queries That Work

Use the `scripts/search_arxiv.py` helper from the `research/arxiv` skill:

```bash
# Search by category + keywords
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py "work stealing distributed" --category cs.DC --max 10 --sort date

# Search by specific arXiv ID
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py --id 2401.04494

# Search by author
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py --author "Setareh Maghsudi" --max 5
```

Direct API queries also work:
```bash
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.DC+AND+all:decentralized+load+balancing&max_results=10&sortBy=submittedDate&sortOrder=descending"
```
