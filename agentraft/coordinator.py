"""The Coordinator — AgentRaft's consensus loop.

For each step it runs the agent, verifies the output, and only *commits* a
checkpoint once the verifier passes it. On failure it rolls back to the last good
checkpoint and retries with a typed correction hint, guarded by a circuit breaker.
This is the Raft-inspired core: nothing is committed without consensus (here, the
verifier standing in for a quorum), and the log can always be rolled back.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .breaker import CircuitBreaker, RetryPolicy
from .checkpoint.base import Checkpoint, CheckpointStore
from .config import AgentRaftConfig
from .errors import CircuitBreakerOpen, MaxRetriesExceeded, VerificationResult
from .events import Event, EventType
from .pipeline import Pipeline, StepContext, Task
from .telemetry import span
from .verifier.base import Verifier, VerifyInput


@dataclass
class StepRun:
    """Trace record for a single step within a run."""

    index: int
    name: str
    status: str  # "committed" | "failed"
    attempts: int
    verification: Optional[VerificationResult]
    elapsed_ms: float
    output: Any = None


@dataclass
class RunResult:
    """Outcome of running a pipeline through the protocol."""

    task_id: str
    success: bool
    output: Any
    steps: list[StepRun] = field(default_factory=list)
    rollbacks: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    @property
    def verified_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "committed")

    @property
    def reliability(self) -> float:
        """Fraction of steps that ended committed (1.0 on full success)."""
        return self.verified_count / len(self.steps) if self.steps else 0.0

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "verified": f"{self.verified_count}/{len(self.steps)}",
            "rollbacks": self.rollbacks,
            "reliability": round(self.reliability, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


class Coordinator:
    """Runs a pipeline under the AgentRaft protocol."""

    def __init__(self, pipeline: Pipeline, config: Optional[AgentRaftConfig] = None):
        from .checkpoint.memory import InMemoryCheckpointStore
        from .verifier.rules import RulesVerifier

        self.pipeline = pipeline
        cfg = config or AgentRaftConfig()
        self.verifier: Verifier = cfg.verifier or RulesVerifier()
        self.store: CheckpointStore = cfg.store or InMemoryCheckpointStore()
        self.retry: RetryPolicy = cfg.retry
        self.breaker: CircuitBreaker = cfg.breaker
        self.config = cfg

    def _emit(self, event: Event) -> None:
        if self.config.on_event:
            try:
                self.config.on_event(event)
            except Exception:
                pass  # never let an event hook break a run

    async def run(self, task: Task) -> RunResult:
        """Execute the pipeline for ``task``, returning a full :class:`RunResult`."""
        t0 = time.monotonic()
        await self.store.clear(task.id)
        result = RunResult(task_id=task.id, success=False, output=None)
        self._emit(Event(EventType.RUN_START, task.id, detail={"goal": task.goal}))

        outputs: dict[str, Any] = {}

        with span("agentraft.run", task_id=task.id, goal=task.goal):
            for index, step in enumerate(self.pipeline.steps):
                step_max_retries = (
                    step.max_retries if step.max_retries is not None else self.retry.max_retries
                )
                attempt = 0
                hint: Optional[str] = None
                last_verification: Optional[VerificationResult] = None
                step_t0 = time.monotonic()

                while True:
                    # Circuit breaker gate.
                    try:
                        self.breaker.check()
                    except CircuitBreakerOpen as e:
                        self._emit(Event(EventType.BREAKER_OPEN, task.id, index, step.name,
                                          attempt, detail=str(e)))
                        result.error = str(e)
                        result.elapsed_ms = (time.monotonic() - t0) * 1000
                        result.steps.append(StepRun(index, step.name, "failed", attempt,
                                                    last_verification,
                                                    (time.monotonic() - step_t0) * 1000))
                        self._emit(Event(EventType.RUN_FAILED, task.id, index, step.name,
                                          detail=str(e)))
                        return result

                    # Backoff between retries.
                    delay = self.retry.delay_for(attempt)
                    if delay:
                        await asyncio.sleep(delay)

                    # 1) Run the step.
                    self._emit(Event(EventType.STEP_START, task.id, index, step.name, attempt))
                    ctx = StepContext(task=task, outputs=dict(outputs), attempt=attempt, hint=hint)
                    with span("agentraft.step", step=step.name, attempt=attempt):
                        output = await step.execute(ctx)

                    # 2) Verify the output.
                    self._emit(Event(EventType.STEP_VERIFYING, task.id, index, step.name, attempt))
                    history = list(outputs.items())
                    with span("agentraft.verify", step=step.name):
                        verification = await self.verifier.verify(
                            VerifyInput(task=task, step=step, output=output,
                                        history=history, attempt=attempt)
                        )
                    last_verification = verification

                    # 3a) Pass -> commit checkpoint.
                    if verification.passed:
                        outputs[step.name] = output
                        await self.store.save(Checkpoint(
                            task_id=task.id, step_index=index, step_name=step.name,
                            output=output, attempt=attempt, verification=verification.to_dict(),
                        ))
                        self.breaker.record_success()
                        self._emit(Event(EventType.STEP_COMMITTED, task.id, index, step.name,
                                         attempt, verification))
                        result.steps.append(StepRun(
                            index, step.name, "committed", attempt + 1, verification,
                            (time.monotonic() - step_t0) * 1000, output,
                        ))
                        break

                    # 3b) Fail -> rollback + retry (within budget).
                    self.breaker.record_failure()
                    self._emit(Event(EventType.STEP_FAILED, task.id, index, step.name,
                                     attempt, verification))

                    if self.config.rollback_on_failure:
                        await self.store.rollback_to(task.id, index - 1)
                        result.rollbacks += 1
                        self._emit(Event(EventType.STEP_ROLLBACK, task.id, index, step.name,
                                         attempt, detail={"to_index": index - 1}))

                    if attempt >= step_max_retries:
                        result.error = (
                            f"Step '{step.name}' exhausted {step_max_retries} retr"
                            f"{'y' if step_max_retries == 1 else 'ies'} "
                            f"(last: {verification.error_type.value})"
                        )
                        result.steps.append(StepRun(
                            index, step.name, "failed", attempt + 1, verification,
                            (time.monotonic() - step_t0) * 1000,
                        ))
                        result.elapsed_ms = (time.monotonic() - t0) * 1000
                        self._emit(Event(EventType.RUN_FAILED, task.id, index, step.name,
                                         detail=result.error))
                        return result

                    # Prepare the retry with the typed correction hint.
                    hint = verification.hint
                    attempt += 1
                    self._emit(Event(EventType.STEP_RETRY, task.id, index, step.name, attempt,
                                     detail={"hint": hint}))

        result.success = True
        result.output = outputs[self.pipeline.steps[-1].name]
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._emit(Event(EventType.RUN_SUCCESS, task.id, detail=result.summary()))
        return result

    async def run_or_raise(self, task: Task) -> Any:
        """Run and return the final output, raising on failure."""
        result = await self.run(task)
        if not result.success:
            last = result.steps[-1].verification if result.steps else None
            raise MaxRetriesExceeded(
                result.steps[-1].name if result.steps else "?",
                result.steps[-1].attempts if result.steps else 0,
                last,
            )
        return result.output
