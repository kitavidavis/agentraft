"""Measure a *real* verifier's quality, per error class.

This is the bridge to the moat. Given any verifier (rules, OpenAI, Anthropic,
Bedrock, Gemini, or a future fine-tuned model), it generates known-bad and
known-good outputs and measures:

  - recall per error class : of the bad outputs we injected, how many did it catch?
  - false-positive rate     : of the good outputs, how many did it wrongly flag?
  - class accuracy          : when it caught a fault, did it name the right class?

These are exactly the numbers you'd track while training/selecting a verifier, and
they explain *why* a semantic verifier beats a rules gate on the harder classes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentraft import Criticality, ErrorType, Step, Task
from agentraft.verifier.base import Verifier, VerifyInput

from .faults import _render

_BAD_CLASSES = [
    ErrorType.GOAL_DRIFT,
    ErrorType.HALLUCINATION,
    ErrorType.INCOMPLETE,
    ErrorType.CONTRADICTION,
    ErrorType.SCOPE_CREEP,
]


@dataclass
class ClassStat:
    total: int = 0
    caught: int = 0          # verifier flagged it (any class)
    correct_class: int = 0   # verifier flagged it with the right class

    @property
    def recall(self) -> float:
        return self.caught / self.total if self.total else 0.0

    @property
    def class_accuracy(self) -> float:
        return self.correct_class / self.caught if self.caught else 0.0


@dataclass
class ConfusionResult:
    verifier_name: str
    per_class: dict[ErrorType, ClassStat] = field(default_factory=dict)
    good_total: int = 0
    good_flagged: int = 0  # false positives

    @property
    def false_positive_rate(self) -> float:
        return self.good_flagged / self.good_total if self.good_total else 0.0

    @property
    def overall_recall(self) -> float:
        tot = sum(s.total for s in self.per_class.values())
        caught = sum(s.caught for s in self.per_class.values())
        return caught / tot if tot else 0.0


def _make_input(label: ErrorType, i: int) -> VerifyInput:
    name = f"stage_{i:02d}"
    # A prior verified output so CONTRADICTION has something to contradict.
    history = [("prior_step", "The established conclusion is that revenue grew 12% in Q3.")]
    step = Step(name, lambda ctx: "", goal=f"Produce a correct result for {name}",
                criticality=Criticality.HIGH)
    return VerifyInput(
        task=Task(goal="Produce an accurate multi-step analysis", id=f"conf-{i}"),
        step=step,
        output=_render(name, 0, label),
        history=history,
        attempt=0,
    )


async def measure_verifier(
    verifier: Verifier, *, samples_per_class: int = 20, seed: int = 7
) -> ConfusionResult:
    """Probe a verifier with known-bad and known-good outputs and tally its accuracy."""
    rng = random.Random(seed)
    res = ConfusionResult(verifier_name=getattr(verifier, "name", type(verifier).__name__))

    for cls in _BAD_CLASSES:
        stat = ClassStat()
        for i in range(samples_per_class):
            verdict = await verifier.verify(_make_input(cls, rng.randint(0, 10_000)))
            stat.total += 1
            if not verdict.passed:
                stat.caught += 1
                if verdict.error_type is cls:
                    stat.correct_class += 1
        res.per_class[cls] = stat

    # Good outputs -> measure false positives.
    for i in range(samples_per_class):
        verdict = await verifier.verify(_make_input(ErrorType.NONE, rng.randint(0, 10_000)))
        res.good_total += 1
        if not verdict.passed:
            res.good_flagged += 1

    return res
