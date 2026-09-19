"""GuessingModel interface + per-node state — Phase 1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from simulation.types import Capability, Task


@dataclass
class GuessingModel:
    """Per-node local cost estimator with gossip support.

    Phase 1: stub that returns a constant estimate.  Phase 2+ adds
    actual learning and gossip merge/snapshot.
    """

    node_id: str
    capability: Capability

    def estimate(self, task: Task) -> float:
        """Estimate cost of *task* on this node."""
        cpu_ratio = task.cpu_req / max(self.capability.cpu, 1e-9)
        mem_ratio = task.memory_req / max(self.capability.memory, 1e-9)
        net_ratio = task.network_req / max(self.capability.network, 1e-9)
        return max(cpu_ratio, mem_ratio, net_ratio) * 10.0

    def snapshot(self) -> dict[str, Any]:
        """Return state for gossip exchange."""
        return {
            "node_id": self.node_id,
            "capability": {
                "cpu": self.capability.cpu,
                "memory": self.capability.memory,
                "network": self.capability.network,
            },
        }

    def merge(self, other: dict[str, Any]) -> None:
        """Ingest a gossip snapshot from a peer node."""
        # Phase 1: no-op; Phase 2+ implements merging logic.
        _ = other  # noqa: F841