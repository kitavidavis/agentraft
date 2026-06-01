"""A simulated verifier with tunable recall and false-positive rate.

This lets us sweep verifier *quality* and see how end-to-end reliability depends on
it — the core argument for investing in a better (fine-tuned) verifier. It reads the
ground-truth ledger, then applies the configured detection probabilities.

  recall = P(flag | output is actually bad)        — true-positive rate
  fpr    = P(flag | output is actually good)        — false-positive rate

A perfect verifier is recall=1.0, fpr=0.0. Real verifiers live below that, which is
exactly what the benchmark quantifies in ``--live`` mode.
"""
from __future__ import annotations

import random

from agentraft import ErrorType, VerificationResult
from agentraft.verifier.base import Verifier, VerifyInput

from .faults import GroundTruth

_FP_CLASSES = [
    ErrorType.GOAL_DRIFT,
    ErrorType.HALLUCINATION,
    ErrorType.INCOMPLETE,
    ErrorType.CONTRADICTION,
    ErrorType.SCOPE_CREEP,
]


class SimulatedVerifier(Verifier):
    """Ground-truth-aware verifier parameterised by recall and false-positive rate."""

    def __init__(self, *, recall: float, fpr: float, ledger: GroundTruth, rng: random.Random):
        self.recall = recall
        self.fpr = fpr
        self.ledger = ledger
        self.rng = rng
        self.name = f"sim(recall={recall:.2f},fpr={fpr:.2f})"

    async def verify(self, inp: VerifyInput) -> VerificationResult:
        truth = self.ledger.get(inp.step.name, inp.attempt)

        if truth is not ErrorType.NONE:
            # Output is genuinely bad — catch it with probability=recall.
            if self.rng.random() < self.recall:
                return VerificationResult.fail(truth, reasoning="sim detected", verifier=self.name)
            return VerificationResult.ok(verifier=self.name, reasoning="sim miss")

        # Output is genuinely good — false-positive with probability=fpr.
        if self.rng.random() < self.fpr:
            cls = self.rng.choice(_FP_CLASSES)
            return VerificationResult.fail(cls, reasoning="sim false positive", verifier=self.name)
        return VerificationResult.ok(verifier=self.name, reasoning="sim pass")
