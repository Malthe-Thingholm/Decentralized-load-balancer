"""Node dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Node:
    """A compute node with capabilities and a task queue."""

    id: str
    cpu_cap: float
    memory_cap: float
    network_cap: float
    queue: list[str] = field(default_factory=list)  # task ids
    completed: list[str] = field(default_factory=list)
    total_busy_time: float = 0.0
    current_load_end: float = 0.0  # sim-time when current queue clears
    guessing_model_state: dict = field(default_factory=dict)
    on_complete: Callable | None = None  # completion callback hook

    def __post_init__(self) -> None:
        if self.cpu_cap <= 0:
            raise ValueError(f"cpu_cap must be > 0, got {self.cpu_cap}")
        if self.memory_cap <= 0:
            raise ValueError(f"memory_cap must be > 0, got {self.memory_cap}")
        if self.network_cap <= 0:
            raise ValueError(f"network_cap must be > 0, got {self.network_cap}")

    @property
    def is_busy(self) -> bool:
        return len(self.queue) > 0

    @property
    def queue_depth(self) -> int:
        return len(self.queue)

    def assign(self, task_id: str) -> None:
        self.queue.append(task_id)

    def complete(self, task_id: str, sim_time: float) -> float:
        """Mark a task complete, return its service time."""
        if task_id in self.queue:
            self.queue.remove(task_id)
        service_time = sim_time - self._arrival_times.get(task_id, sim_time)
        self.completed.append(task_id)
        self.total_busy_time += service_time
        if self.on_complete is not None:
            self.on_complete(task_id, self.id, service_time)
        return service_time

    # Internal tracking for service time calculation
    _arrival_times: dict[str, float] = field(default_factory=dict, repr=False)

    def record_arrival(self, task_id: str, sim_time: float) -> None:
        self._arrival_times[task_id] = sim_time