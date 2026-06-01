"""Redis-backed checkpoint store for durable, multi-process pipelines.

Each task's checkpoints live in a Redis list keyed by ``{prefix}:{task_id}`` and
serialized as JSON. Requires the ``redis`` extra: ``pip install agentraft[redis]``.
"""
from __future__ import annotations

import json
from typing import Optional

from ..errors import AgentRaftError
from .base import Checkpoint, CheckpointStore


class RedisCheckpointStore(CheckpointStore):
    """Durable checkpoint store using a Redis list per task."""

    def __init__(self, url: str = "redis://localhost:6379/0", *, prefix: str = "agentraft:ckpt"):
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise AgentRaftError(
                "RedisCheckpointStore requires the 'redis' extra. "
                "Install with: pip install agentraft[redis]"
            ) from e
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}:{task_id}"

    async def save(self, cp: Checkpoint) -> None:
        await self._redis.rpush(self._key(cp.task_id), json.dumps(cp.to_dict()))

    async def latest(self, task_id: str) -> Optional[Checkpoint]:
        raw = await self._redis.lindex(self._key(task_id), -1)
        return Checkpoint.from_dict(json.loads(raw)) if raw else None

    async def at(self, task_id: str, step_index: int) -> Optional[Checkpoint]:
        for cp in reversed(await self.history(task_id)):
            if cp.step_index == step_index:
                return cp
        return None

    async def history(self, task_id: str) -> list[Checkpoint]:
        raw = await self._redis.lrange(self._key(task_id), 0, -1)
        return [Checkpoint.from_dict(json.loads(r)) for r in raw]

    async def rollback_to(self, task_id: str, step_index: int) -> None:
        kept = [cp for cp in await self.history(task_id) if cp.step_index <= step_index]
        key = self._key(task_id)
        pipe = self._redis.pipeline()
        pipe.delete(key)
        for cp in kept:
            pipe.rpush(key, json.dumps(cp.to_dict()))
        await pipe.execute()

    async def clear(self, task_id: str) -> None:
        await self._redis.delete(self._key(task_id))

    async def close(self) -> None:
        await self._redis.aclose()
