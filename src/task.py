"""Task dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """A computational task with resource requirements."""

    id: str
    cpu_req: float
    memory_req: float
    network_req: float
    deadline: float | None = None
    priority: int = 0
    arrival_time: float = 0.0

    def __post_init__(self) -> None:
        if self.cpu_req < 0:
            raise ValueError(f"cpu_req must be >= 0, got {self.cpu_req}")
        if self.memory_req < 0:
            raise ValueError(f"memory_req must be >= 0, got {self.memory_req}")
        if self.network_req < 0:
            raise ValueError(f"network_req must be >= 0, got {self.network_req}")