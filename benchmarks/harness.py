"""The benchmark engine: run faulty pipelines with and without AgentRaft, measure lift.

Three outcomes per protected trial:
  SUCCESS           — completed, every committed output correct.
  SILENT_CORRUPTION — completed, but a bad output slipped through (verifier miss). The
                      dangerous case: a wrong result shipped with full confidence.
  CAUGHT_FAILURE    — failed safely; a bad step was flagged but couldn't be recovered
                      within the retry budget. Not a success, but nothing wrong shipped.

The baseline (no AgentRaft) has only SUCCESS or SILENT_CORRUPTION — it can't catch
anything, so every corrupted run ships silently.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agentraft import (
    ErrorType,
    InMemoryCheckpointStore,
    StepContext,
    Task,
    wrap,
)
from agentraft.verifier.base import Verifier

from .faults import FaultConfig, GroundTruth, build_pipeline
from .sim_verifier import SimulatedVerifier

_VSEED_SALT = 0x9E3779B9  # de-correlate the verifier RNG from the agent RNG


class Outcome(str, Enum):
    SUCCESS = "success"
    SILENT_CORRUPTION = "silent_corruption"
    CAUGHT_FAILURE = "caught_failure"


@dataclass
class Scenario:
    """One benchmark configuration."""

    n_steps: int = 10
    fault: FaultConfig = field(default_factory=FaultConfig)
    max_retries: int = 3
    # Simulated verifier quality (ignored in live mode).
    recall: float = 0.9
    fpr: float = 0.03
    # Relative cost of one verification vs one generation (verification asymmetry).
    verify_cost_ratio: float = 0.08
    breaker_threshold: int = 10_000  # effectively off, to isolate step-level recovery
    label: str = ""


@dataclass
class Aggregate:
    trials: int
    success: int = 0
    silent_corruption: int = 0
    caught_failure: int = 0
    total_generations: int = 0
    total_verifications: int = 0
    total_rollbacks: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.trials if self.trials else 0.0

    @property
    def silent_corruption_rate(self) -> float:
        return self.silent_corruption / self.trials if self.trials else 0.0

    @property
    def caught_failure_rate(self) -> float:
        return self.caught_failure / self.trials if self.trials else 0.0

    def record(self, outcome: Outcome) -> None:
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)


@dataclass
class Comparison:
    scenario: Scenario
    baseline: Aggregate
    protected: Aggregate
    # cost: average "compute units" per task (1 unit = 1 generation).
    baseline_cost: float = 0.0
    protected_cost: float = 0.0
    full_rerun_cost: float = 0.0  # naive "rerun the whole pipeline on any failure" strategy

    @property
    def corruption_reduction(self) -> float:
        """How much AgentRaft cuts silent corruption vs baseline (x-factor)."""
        b = self.baseline.silent_corruption_rate
        p = self.protected.silent_corruption_rate
        if p <= 0:
            return float("inf") if b > 0 else 1.0
        return b / p

    @property
    def cost_vs_full_rerun(self) -> float:
        """How many times cheaper AgentRaft is than naive full-pipeline reruns."""
        return self.full_rerun_cost / self.protected_cost if self.protected_cost else 0.0


async def _run_baseline_trial(scenario: Scenario, seed: int) -> Outcome:
    """Run the pipeline once, no verification — corruption ships silently."""
    rng = random.Random(seed)
    ledger = GroundTruth()
    pipeline = build_pipeline(scenario.n_steps, scenario.fault, rng, ledger)
    task = Task(goal="benchmark task", id=f"base-{seed}")

    outputs: dict[str, str] = {}
    for step in pipeline.steps:
        ctx = StepContext(task=task, outputs=dict(outputs), attempt=0)
        outputs[step.name] = await step.execute(ctx)

    corrupted = any(ledger.get(s.name, 0) is not ErrorType.NONE for s in pipeline.steps)
    return Outcome.SILENT_CORRUPTION if corrupted else Outcome.SUCCESS


async def _run_protected_trial(
    scenario: Scenario, seed: int, verifier: Optional[Verifier]
) -> tuple[Outcome, int, int]:
    """Run the pipeline under AgentRaft. Returns (outcome, generations, rollbacks)."""
    rng = random.Random(seed)  # same seed as baseline -> identical attempt-0 faults
    ledger = GroundTruth()
    pipeline = build_pipeline(scenario.n_steps, scenario.fault, rng, ledger)
    task = Task(goal="benchmark task", id=f"prot-{seed}")

    v = verifier or SimulatedVerifier(
        recall=scenario.recall, fpr=scenario.fpr, ledger=ledger,
        rng=random.Random(seed ^ _VSEED_SALT),
    )
    store = InMemoryCheckpointStore()
    coord = wrap(
        pipeline, verifier=v, store=store,
        max_retries=scenario.max_retries, failure_threshold=scenario.breaker_threshold,
    )
    result = await coord.run(task)

    generations = sum(sr.attempts for sr in result.steps)
    committed = await store.history(task.id)
    bad_committed = any(ledger.get(cp.step_name, cp.attempt) is not ErrorType.NONE for cp in committed)

    if result.success and not bad_committed:
        outcome = Outcome.SUCCESS
    elif result.success and bad_committed:
        outcome = Outcome.SILENT_CORRUPTION  # verifier missed a real fault
    else:
        outcome = Outcome.CAUGHT_FAILURE

    return outcome, generations, result.rollbacks


async def evaluate(
    scenario: Scenario,
    *,
    trials: int = 400,
    seed: int = 1234,
    verifier: Optional[Verifier] = None,
) -> Comparison:
    """Run ``trials`` paired baseline/protected trials and aggregate."""
    base = Aggregate(trials=trials)
    prot = Aggregate(trials=trials)

    for t in range(trials):
        s = seed + t
        base.record(await _run_baseline_trial(scenario, s))

        outcome, gens, rollbacks = await _run_protected_trial(scenario, s, verifier)
        prot.record(outcome)
        prot.total_generations += gens
        prot.total_verifications += gens  # one verification per generation
        prot.total_rollbacks += rollbacks

    # ── Cost model (compute units; 1 unit = 1 generation) ────────────────────
    n = scenario.n_steps
    baseline_cost = float(n)  # one generation per step, no verification
    avg_gen = prot.total_generations / trials
    avg_verif = prot.total_verifications / trials
    protected_cost = avg_gen + avg_verif * scenario.verify_cost_ratio

    # Naive alternative: rerun the WHOLE pipeline until it succeeds (no step granularity).
    p_ok_pipeline = (1.0 - scenario.fault.p_fail) ** n
    expected_full_runs = 1.0 / p_ok_pipeline if p_ok_pipeline > 0 else float("inf")
    full_rerun_cost = n * expected_full_runs

    return Comparison(
        scenario=scenario, baseline=base, protected=prot,
        baseline_cost=baseline_cost, protected_cost=protected_cost,
        full_rerun_cost=full_rerun_cost,
    )


def evaluate_sync(scenario: Scenario, **kw) -> Comparison:
    return asyncio.run(evaluate(scenario, **kw))
