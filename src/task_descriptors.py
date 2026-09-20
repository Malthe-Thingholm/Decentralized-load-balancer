"""Task descriptors — type, size, deadline, dependencies.

Each Task is a unit of work routed through the load balancer.
Dependencies are IDs of other tasks that must complete first.
"""

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    task_type: str          # e.g. "compute", "io", "mixed"
    size: float             # work units required
    deadline: float         # simulation-clock timestamp by which it must finish
    dependencies: list[str] = field(default_factory=list)  # task IDs that must complete first

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """True when all dependency tasks have completed."""
        return all(dep in completed_tasks for dep in self.dependencies)


if __name__ == "__main__":
    t_a = Task(task_id="t-1", task_type="compute", size=4.0, deadline=100.0)
    t_b = Task(task_id="t-2", task_type="io", size=2.0, deadline=150.0, dependencies=["t-1"])

    done = {"t-1"}
    for t in (t_a, t_b):
        print(f"{t.task_id}: type={t.task_type}, size={t.size}, "
              f"ready={t.is_ready(done)}")
