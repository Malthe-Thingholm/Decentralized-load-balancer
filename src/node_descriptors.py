"""Node descriptors — identity, capacity, location, current load.

Each Node represents a compute unit in the decentralized system:
where it is, how much it can handle, and how loaded it currently is.
"""

from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: str
    capacity: float  # total processing units (e.g., normalized 0–1 or vCPUs)
    location: tuple[float, float]  # (lat, lon) for network-distance modeling
    current_load: float = field(default=0.0)  # currently assigned workload

    @property
    def utilization(self) -> float:
        """Current load as a fraction of capacity (0.0 – inf)."""
        if self.capacity <= 0:
            return 0.0
        return self.current_load / self.capacity

    def available(self) -> float:
        """Remaining capacity after current load."""
        return max(0.0, self.capacity - self.current_load)


if __name__ == "__main__":
    n1 = Node(node_id="n-alpha", capacity=10.0, location=(37.77, -122.42))
    n2 = Node(node_id="n-beta", capacity=5.0, location=(34.05, -118.24), current_load=3.0)

    for n in (n1, n2):
        print(f"{n.node_id}: capacity={n.capacity}, load={n.current_load}, "
              f"util={n.utilization:.2f}, available={n.available()}")
