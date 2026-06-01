"""AgentRaft — distributed reliability infrastructure for agentic AI.

Wrap any multi-step agent pipeline with Raft-inspired step-level consensus:
every step is verified before it commits, and the pipeline rolls back to the last
good checkpoint on failure.

    from agentraft import wrap, Pipeline, Task

    pipeline = Pipeline([research, analyse, draft, review, publish])
    result = await wrap(pipeline).run(Task(goal="Write the Q3 board memo"))

By default ``wrap`` builds a tiered verifier (L1 rules + an LLM verifier if an
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` is present), an in-memory checkpoint
store (or Redis if ``REDIS_URL`` is set), and a circuit breaker.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from .breaker import BreakerState, CircuitBreaker, RetryPolicy
from .checkpoint import Checkpoint, CheckpointStore, InMemoryCheckpointStore
from .config import AgentRaftConfig
from .coordinator import Coordinator, RunResult, StepRun
from .errors import (
    AgentRaftError,
    CircuitBreakerOpen,
    CORRECTION_HINTS,
    ErrorType,
    MaxRetriesExceeded,
    VerificationResult,
    VerifierUnavailable,
)
from .events import Event, EventType
from .pipeline import Criticality, Pipeline, Step, StepContext, Task, step
from .verifier import MockVerifier, RulesVerifier, TieredVerifier, Verifier, VerifyInput

__version__ = "0.1.0"

__all__ = [
    "wrap",
    "Pipeline", "Step", "StepContext", "Task", "Criticality", "step",
    "Coordinator", "RunResult", "StepRun", "AgentRaftConfig",
    "Verifier", "VerifyInput", "RulesVerifier", "TieredVerifier", "MockVerifier",
    "CheckpointStore", "InMemoryCheckpointStore", "Checkpoint",
    "CircuitBreaker", "RetryPolicy", "BreakerState",
    "ErrorType", "VerificationResult", "CORRECTION_HINTS",
    "Event", "EventType",
    "AgentRaftError", "CircuitBreakerOpen", "MaxRetriesExceeded", "VerifierUnavailable",
    "__version__",
]


def _llm_provider_configured() -> bool:
    """True if any LLM verifier provider can be auto-detected from the environment."""
    return bool(
        os.getenv("AGENTRAFT_VERIFIER_PROVIDER")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        # Amazon Bedrock — the common enterprise path (AWS creds / profile / role).
        or os.getenv("AGENTRAFT_USE_BEDROCK")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
    )


def _default_verifier() -> Verifier:
    """Build the default verifier: L1 rules, escalating to an LLM if one is configured.

    Provider is auto-detected (see ``LLMVerifier._detect_provider``): Bedrock,
    OpenAI, Anthropic, or Gemini.
    """
    l1 = RulesVerifier()
    if _llm_provider_configured():
        try:
            from .verifier.llm import LLMVerifier

            return TieredVerifier(l1=l1, l2=LLMVerifier())
        except Exception:
            pass  # fall back to rules-only if provider import/config fails
    return TieredVerifier(l1=l1)


def _default_store() -> CheckpointStore:
    """Use Redis if REDIS_URL is set, otherwise an in-memory store."""
    url = os.getenv("REDIS_URL")
    if url:
        try:
            from .checkpoint.redis_store import RedisCheckpointStore

            return RedisCheckpointStore(url)
        except Exception:
            pass
    return InMemoryCheckpointStore()


def wrap(
    pipeline: Pipeline,
    *,
    verifier: Optional[Verifier] = None,
    store: Optional[CheckpointStore] = None,
    max_retries: int = 2,
    failure_threshold: int = 5,
    cooldown_seconds: float = 30.0,
    rollback_on_failure: bool = True,
    on_event: Optional[Callable[[Event], None]] = None,
    config: Optional[AgentRaftConfig] = None,
) -> Coordinator:
    """Wrap a pipeline with the AgentRaft protocol and return a runnable Coordinator.

    Pass ``config`` to supply a fully-built :class:`AgentRaftConfig`, or use the
    keyword shortcuts to tweak the common knobs.
    """
    if config is None:
        config = AgentRaftConfig(
            verifier=verifier or _default_verifier(),
            store=store or _default_store(),
            retry=RetryPolicy(max_retries=max_retries),
            breaker=CircuitBreaker(
                failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds
            ),
            rollback_on_failure=rollback_on_failure,
            on_event=on_event,
        )
    return Coordinator(pipeline, config)
