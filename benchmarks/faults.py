"""Controlled fault injection for the AgentRaft benchmark.

We model an agent pipeline where each step is *faulty* with a tunable probability
and, when it fails, emits an output corrupted in one of the five taxonomy classes.
Because we inject the fault, we know the ground-truth label of every output — which
is what lets us measure verifier recall and end-to-end silent-corruption rate exactly.

Outputs are realistic text (not just a label), so the *same* faulty pipeline can be
judged by either the simulated verifier (ground-truth aware) or a real LLM/rules
verifier in ``--live`` mode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentraft import Criticality, ErrorType, Pipeline, Step, StepContext

# Default mix of error classes when a step fails.
DEFAULT_ERROR_DIST: dict[ErrorType, float] = {
    ErrorType.GOAL_DRIFT: 0.30,
    ErrorType.HALLUCINATION: 0.25,
    ErrorType.INCOMPLETE: 0.20,
    ErrorType.CONTRADICTION: 0.15,
    ErrorType.SCOPE_CREEP: 0.10,
}


class GroundTruth:
    """Per-trial ledger mapping ``(step_name, attempt) -> injected ErrorType``."""

    def __init__(self) -> None:
        self._labels: dict[tuple[str, int], ErrorType] = {}

    def set(self, step_name: str, attempt: int, label: ErrorType) -> None:
        self._labels[(step_name, attempt)] = label

    def get(self, step_name: str, attempt: int) -> ErrorType:
        return self._labels.get((step_name, attempt), ErrorType.NONE)


def _choose_error(dist: dict[ErrorType, float], rng: random.Random) -> ErrorType:
    classes = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(classes, weights=weights, k=1)[0]


def _render(step_name: str, attempt: int, label: ErrorType) -> str:
    """Produce realistic output text for a (correct | corrupted) step result."""
    topic = step_name.replace("_", " ")
    if label is ErrorType.NONE:
        return (
            f"[{step_name}] Completed the required work for the {topic} stage: produced "
            f"a correct, complete result consistent with the prior steps and the task goal."
        )
    return {
        ErrorType.GOAL_DRIFT: (
            f"[{step_name}] Here are some unrelated marketing taglines and fun facts — "
            f"none of which address the {topic} task that was actually requested."
        ),
        ErrorType.HALLUCINATION: (
            f"[{step_name}] According to the (fabricated) 2025 Gartner report #99-Z, the "
            f"figure is exactly 47.3% — a statistic that appears in none of the inputs."
        ),
        ErrorType.INCOMPLETE: f"[{step_name}] Partial result for the {topic} stage — the analysis stops abru",
        ErrorType.CONTRADICTION: (
            f"[{step_name}] This result directly reverses the conclusion that the previous "
            f"verified step established, asserting the opposite without justification."
        ),
        ErrorType.SCOPE_CREEP: (
            f"[{step_name}] Done — and I also went ahead and deleted the old records, emailed "
            f"the client, and changed the pricing, none of which were in scope for this step."
        ),
    }[label]


@dataclass
class FaultConfig:
    """Knobs controlling how faulty the simulated agents are."""

    p_fail: float = 0.10  # base per-step failure probability (1 - per-step reliability)
    error_dist: dict[ErrorType, float] = field(default_factory=lambda: dict(DEFAULT_ERROR_DIST))
    retry_improvement: float = 0.45  # failure prob is multiplied by this on each retry (hints help)


def make_faulty_step(
    name: str,
    cfg: FaultConfig,
    rng: random.Random,
    ledger: GroundTruth,
    *,
    criticality: Criticality = Criticality.HIGH,
) -> Step:
    """Build a Step whose output is correct or corrupted according to ``cfg`` and ``rng``."""

    async def fn(ctx: StepContext) -> str:
        attempt = ctx.attempt
        # Typed correction hints make retries more likely to succeed.
        eff_p_fail = cfg.p_fail * (cfg.retry_improvement ** attempt)
        if rng.random() < eff_p_fail:
            label = _choose_error(cfg.error_dist, rng)
        else:
            label = ErrorType.NONE
        ledger.set(name, attempt, label)
        return _render(name, attempt, label)

    return Step(name, fn, goal=f"Produce a correct, in-scope result for the {name} stage",
                criticality=criticality)


def build_pipeline(
    n_steps: int,
    cfg: FaultConfig,
    rng: random.Random,
    ledger: GroundTruth,
    *,
    criticality: Criticality = Criticality.HIGH,
) -> Pipeline:
    """Build an n-step faulty pipeline sharing one RNG + ground-truth ledger."""
    steps = [
        make_faulty_step(f"step_{i:02d}", cfg, rng, ledger, criticality=criticality)
        for i in range(1, n_steps + 1)
    ]
    return Pipeline(steps, name=f"faulty_{n_steps}step")
