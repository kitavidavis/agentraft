"""Configuration for a wrapped pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .breaker import CircuitBreaker, RetryPolicy
from .checkpoint.base import CheckpointStore
from .verifier.base import Verifier


@dataclass
class AgentRaftConfig:
    """Knobs for the Coordinator. Sensible defaults are filled in by :func:`agentraft.wrap`."""

    verifier: Optional[Verifier] = None
    store: Optional[CheckpointStore] = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    #: Optional event hook — receives protocol events (see events.py). Used to
    #: drive live monitors/dashboards. Must be cheap and non-blocking.
    on_event: Optional[Callable[["object"], None]] = None

    #: If True, a failed verification rolls back the checkpoint store to the prior
    #: step before retrying. If False, retries happen without touching the store.
    rollback_on_failure: bool = True
