"""Checkpoint stores for the AgentRaft protocol."""
from .base import Checkpoint, CheckpointStore
from .memory import InMemoryCheckpointStore

__all__ = ["Checkpoint", "CheckpointStore", "InMemoryCheckpointStore", "RedisCheckpointStore"]


def __getattr__(name: str):
    # Lazy import so importing the package never hard-requires redis.
    if name == "RedisCheckpointStore":
        from .redis_store import RedisCheckpointStore

        return RedisCheckpointStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
