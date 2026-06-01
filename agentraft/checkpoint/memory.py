"""In-memory checkpoint store — the zero-dependency default."""
from __future__ import annotations

import asyncio
from typing import Optional

from .base import Checkpoint, CheckpointStore


class InMemoryCheckpointStore(CheckpointStore):
    """Stores checkpoints in a process-local dict. Great for tests and single-process runs."""

    def __init__(self) -> None:
        self._data: dict[str, list[Checkpoint]] = {}
        self._lock = asyncio.Lock()

    async def save(self, cp: Checkpoint) -> None:
        async with self._lock:
            self._data.setdefault(cp.task_id, []).append(cp)

    async def latest(self, task_id: str) -> Optional[Checkpoint]:
        items = self._data.get(task_id)
        return items[-1] if items else None

    async def at(self, task_id: str, step_index: int) -> Optional[Checkpoint]:
        for cp in reversed(self._data.get(task_id, [])):
            if cp.step_index == step_index:
                return cp
        return None

    async def history(self, task_id: str) -> list[Checkpoint]:
        return list(self._data.get(task_id, []))

    async def rollback_to(self, task_id: str, step_index: int) -> None:
        async with self._lock:
            items = self._data.get(task_id, [])
            self._data[task_id] = [cp for cp in items if cp.step_index <= step_index]

    async def clear(self, task_id: str) -> None:
        async with self._lock:
            self._data.pop(task_id, None)
