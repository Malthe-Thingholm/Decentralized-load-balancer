# Decentralized Load Balancer Papers — Detailed Notes

## Collected from Arxiv (Phase 1 research)

### 1. Adaptive Asynchronous Work-Stealing (A2WS)
- **URL:** https://arxiv.org/abs/2401.04494
- **arXiv ID:** 2401.04494v2
- **Title:** Adaptive Asynchronous Work-Stealing for distributed load-balancing in heterogeneous systems
- **Date:** January 2024 (v2, updated Jan 23)
- **Domain:** cs.DC (Distributed, Parallel, and Cluster Computing)
- **Authors:** João B. Fernandes, Ítalo A. S. de Assis, Idalmis M. S. Martins, Tiago Barros, Samuel Xavier-de-Souza
- **Abstract (summary):** Supercomputers group hundreds/thousands of computing nodes to execute time-consuming programs requiring large computational resources. Over the years, supercomputers have transitioned from monolithic to heterogeneous. This paper presents an adaptive asynchronous work-stealing approach for distributed load-balancing in heterogeneous systems. Key contributions: smart victim selection (which node to steal from), limited information communication (nodes don't need full state), and asynchronous theft (no blocking coordination).
- **Key idea:** Smart victim selection + limited info communication + asynchronous theft. Adapts to heterogeneity without central coordinator.
- **Relevance to sim:** High. Candidate for work-stealing baseline and gossip-based state estimation. Maps directly to: idle nodes "steal" work from busy nodes using partial state information.
- **Implementation notes:** Need to model victim selection heuristic (which overloaded node to target), steal request/reply protocol, adaptation to different node capacities. Gossip can serve as the "limited info communication" channel.

### 2. Recycled Entropy Packet Spraying (REPS)
- **URL:** https://arxiv.org/abs/2407.21625
- **arXiv ID:** 2407.21625
- **Title:** REPS: Adaptive Load Balancing Through Recycled Entropy Packet Spraying
- **Date:** July 2024
- **Domain:** cs.NI (Networking and Internet Architecture)
- **Key idea:** Entropy-based packet spraying that recycles load information to maintain adaptivity. Stochastic routing guided by load signals — entropy serves as a randomness source that gets "recycled" from observed load feedback.
- **Relevance to sim:** Medium-high. Candidate for adaptive probabilistic routing baseline. Tests whether entropy-based routing beats simple random. Minimal state requirement.
- **Implementation notes:** Probabilistic task routing where probabilities adapt based on observed load feedback. Entropy recycling mechanism needs modeling.

### 3. Mean-Field Multi-Armed Bandits (MF-MAB)
- **URL:** https://arxiv.org/abs/2407.00080
- **arXiv ID:** 2407.00080v1
- **Title:** Decentralized Task Offloading and Load-Balancing for Mobile Edge Computing in Dense Networks
- **Date:** June 2024
- **Domain:** cs.DC, cs.LG, cs.MA
- **Authors:** Mariam Yahya, Alexander Conzelmann, Setareh Maghsudi
- **Abstract (summary):** Studies decentralized task offloading and load-balancing in dense networks with numerous devices and edge servers. Optimal solving is complicated by unknown network information and random task sizes. Shared network resources also influence usage patterns. Uses mean-field multi-armed bandit approach where each device acts as a bandit deciding which edge server to offload to; mean-field approximation handles the large population size.
- **Key idea:** Each IoT device acts as a bandit deciding which edge server to offload to; mean-field approximation handles the population size. Solves unknown network info + random task sizes.
- **Relevance to sim:** High. Candidate for bandit-based load balancer. Maps directly to task→node assignment with exploration/exploitation tradeoff.
- **Implementation notes:** Each node runs a bandit algorithm; mean-field term captures aggregate population behavior. UCB-style confidence bounds for exploration. State = observed costs per node; action = which node to assign to.

### 4. QoS-Aware Load Balancing via Multi-Player Bandits (QEdgeProxy)
- **URL:** https://arxiv.org/abs/2512.18915
- **arXiv ID:** 2512.18915
- **Title:** QEdgeProxy: QoS-Aware Load Balancing in Edge Computing via Multi-Player Multi-Armed Bandits
- **Date:** December 2025
- **Domain:** cs.DC, cs.NI
- **Key idea:** Multiple proxy nodes independently select edge servers using bandits; handles collisions when multiple proxies pick the same server. QoS-aware: respects quality-of-service constraints in the bandit reward.
- **Relevance to sim:** Medium. Candidate for multi-agent competitive assignment. Tests collision-avoidance in decentralized selection. QoS constraints map to task deadlines/priorities.
- **Implementation notes:** Multi-player bandit with collision handling. Each proxy independently selects; collision resolution needed when multiple proxies pick the same server. QoS constraints as reward shaping.

### 5. Incentive-Based Decentralized LB
- **URL:** https://arxiv.org/abs/2501.01219
- **arXiv ID:** 2501.01219v1
- **Title:** Model of an Open, Decentralized Computational Network with Incentive-Based Load Balancing
- **Date:** January 2025
- **Domain:** q-fin.CP, math.DS, math.OC
- **Authors:** German Rodikov
- **Abstract (summary):** Proposes a model enabling permissionless and decentralized networks for complex computations. Explores integration and optimizes load balancing in an open, decentralized computational network. Leverages economic incentives and reputation-based mechanisms to dynamically balance load.
- **Key idea:** Load balancing driven by economic incentives (tokens/pricing) rather than pure algorithmic heuristics. Permissionless, decentralized network. Reputation-based mechanisms.
- **Relevance to sim:** Medium (optional, game-theoretic). Candidate for incentive-layer extension. Useful if we want to test economic incentive effects on load balancing behavior.
- **Implementation notes:** Requires modeling token/reputation mechanics. Tasks have costs; nodes have incentives to accept or reject work. Equilibrium analysis needed. May be Phase 4 optional.

---

## Additional Papers from Project Roadmap

### 6. Communication-Efficient DML Balancing
- **URL:** https://arxiv.org/abs/2405.00839
- **arXiv ID:** 2405.00839v1
- **Title:** Communication-Efficient Training Workload Balancing for Decentralized Multi-Agent Learning
- **Date:** May 2024
- **Domain:** cs.LG, cs.AI, cs.DC, cs.MA, cs.PF
- **Authors:** Seyed Mahmoud Sajjadi Mohammadabadi, Lei Yang, Feng Yan, Junshan Zhang
- **Abstract (summary):** Decentralized Multi-agent Learning (DML) enables collaborative model training while preserving data privacy. However, inherent heterogeneity in agents' resources (computation, communication, and task size) may lead to substantial variations in training time, creating bottlenecks. This paper addresses communication-efficient workload balancing for DML.
- **Key idea:** Addresses heterogeneity in agents' resources (computation, communication, task size) that creates bottlenecks in decentralized multi-agent learning. Balances training workload across agents efficiently.
- **Relevance to sim:** High. Heterogeneity-aware balancing; maps to nodes with different capability vectors. Tasks have different sizes; assigns to minimize straggler effect.
- **Implementation notes:** Focus on balancing workload given heterogeneous node capacities and variable task sizes. May involve grouping/partitioning strategies. Communication efficiency aspect may be less relevant for basic load balancer sim.

### 7. RL-Based Adaptive LB for Dynamic Cloud
- **URL:** https://arxiv.org/abs/2409.04896
- **arXiv ID:** 2409.04896v1
- **Title:** Reinforcement Learning-Based Adaptive Load Balancing for Dynamic Cloud Environments
- **Date:** September 2024
- **Domain:** cs.DC, cs.AI, cs.NI
- **Authors:** Kavish Chawla
- **Abstract (summary):** Efficient load balancing is crucial in cloud computing environments to ensure optimal resource utilization, minimize response times, and prevent server overload. Traditional load balancing algorithms (round-robin, least-connections) are often static and unable to adapt to dynamic and fluctuating workloads. This paper presents an RL-based approach for adaptive load balancing.
- **Key idea:** RL agent learns to balance load dynamically. State space includes node loads and queue depths; action is which node to assign to; reward is negative latency/cost. Adapts to changing conditions where static algorithms fail.
- **Relevance to sim:** High. RL-based balancer that learns from observed costs. Maps to: state = node loads + queue depths + recent costs; action = node selection; reward = negative actual cost.
- **Implementation notes:** RL agent design: state space, action space, reward function. Online learning during simulation vs offline pre-training. Exploration vs exploitation. May need simplified RL (tabular Q-learning) before deep RL.

---

## Paper Search Methodology

Use the `scripts/search_arxiv.py` helper from the `research/arxiv` skill:

```bash
# Search by category + keywords
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py "work stealing distributed" --category cs.DC --max 10 --sort date

# Search by specific arXiv ID
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py --id 2401.04494

# Search by author
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py --author "Setareh Maghsudi" --max 5

# Combined query
python /Users/malthe/.hermes/skills/research/arxiv/scripts/search_arxiv.py "decentralized load balancing" --category cs.DC --max 10 --sort date
```

Direct API queries also work:
```bash
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.DC+AND+all:decentralized+load+balancing&max_results=10&sortBy=submittedDate&sortOrder=descending"
```

Semantic Scholar for citations/related papers:
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2401.04494?fields=title,authors,citationCount,referenceCount,abstract" | python -m json.tool
```

## Verified Papers (API-confirmed)

All 7 papers listed above were verified via the arXiv API on 2026-09-20. IDs, titles, authors, and abstracts confirmed.

## Implementation Priority

1. **A2WS work-stealing (2401.04494)** — First adaptive baseline; maps well to existing gossip infrastructure
2. **MF-MAB bandit (2407.00080)** — Natural bandit formulation; good exploration/exploitation testbed
3. **RL-based adaptive LB (2409.04896)** — Phase 3; requires RL infrastructure
4. **Communication-Efficient DML (2405.00839)** — Heterogeneity-aware; complements A2WS
5. **REPS (2407.21625)** — Simple adaptive probabilistic routing; good comparison point
6. **Incentive-Based (2501.01219)** — Optional; game-theoretic extension
7. **QEdgeProxy (2512.18915)** — Optional; multi-agent competition
