"""Checkpoint store abstraction.

A checkpoint is an append-only record of a *verified* step output. The store
powers rollback: on a verification failure, the Coordinator reverts to the last
good checkpoint and replays from there.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Checkpoint:
    """A verified step output, persisted so it can be rolled back to."""

    task_id: str
    step_index: int
    step_name: str
    output: Any
    attempt: int
    verification: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "output": self.output,
            "attempt": self.attempt,
            "verification": self.verification,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(**d)


class CheckpointStore(ABC):
    """Append-only, per-task store of verified checkpoints."""

    @abstractmethod
    async def save(self, cp: Checkpoint) -> None:
        """Persist a checkpoint (must be called only for verified outputs)."""

    @abstractmethod
    async def latest(self, task_id: str) -> Optional[Checkpoint]:
        """Return the most recent checkpoint for a task, or None."""

    @abstractmethod
    async def at(self, task_id: str, step_index: int) -> Optional[Checkpoint]:
        """Return the checkpoint committed at a given step index, or None."""

    @abstractmethod
    async def history(self, task_id: str) -> list[Checkpoint]:
        """Return all checkpoints for a task in commit order."""

    @abstractmethod
    async def rollback_to(self, task_id: str, step_index: int) -> None:
        """Discard every checkpoint *after* ``step_index``.

        Passing ``step_index = -1`` clears the task back to its initial state.
        """

    @abstractmethod
    async def clear(self, task_id: str) -> None:
        """Remove all checkpoints for a task."""

    async def outputs(self, task_id: str) -> dict[str, Any]:
        """Convenience: reconstruct the {step_name: output} map of verified steps."""
        return {cp.step_name: cp.output for cp in await self.history(task_id)}
