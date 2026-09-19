"""Assertion wrapper for TrueModel calls.

Wraps a TrueModel (or stub) and validates:
  - cost is finite and non-negative
  - cost is called exactly once per assignment (configurable)
  - no NaN / inf returned
"""

from __future__ import annotations

import math
import logging
from typing import Any

from sim.true_model import TrueModel, Task, Node

logger = logging.getLogger(__name__)


class AssertionWrapper:
    """Decorator-style wrapper around TrueModel.cost() with assertions."""

    def __init__(self, model: TrueModel, *, max_calls_per_assign: int = 1) -> None:
        self._model = model
        self._max_calls = max_calls_per_assign
        self._call_count: int = 0
        self._errors: list[str] = []

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        self._call_count = 0
        self._errors.clear()

    def cost(self, task: Task, node: Node) -> float:
        """Call model.cost() and assert invariants."""
        self._call_count += 1
        if self._call_count > self._max_calls:
            msg = f"TrueModel called {self._call_count} times (max {self._max_calls})"
            logger.error(msg)
            self._errors.append(msg)
            raise AssertionError(msg)

        result = self._model.cost(task, node)

        if math.isnan(result) or math.isinf(result):
            msg = f"TrueModel returned non-finite value: {result}"
            logger.error(msg)
            self._errors.append(msg)
            raise AssertionError(msg)

        if result < 0:
            msg = f"TrueModel returned negative cost: {result}"
            logger.error(msg)
            self._errors.append(msg)
            raise AssertionError(msg)

        logger.debug("cost(task=%s, node=%s) = %.6f", task.task_id, node.node_id, result)
        return result

    def assert_clean(self) -> None:
        """Raise if any assertion was violated."""
        if self._errors:
            raise AssertionError(f"assertion failures: {self._errors}")
