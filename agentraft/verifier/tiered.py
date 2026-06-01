"""Tiered verifier — routes each step to the cheapest sufficient verifier.

L1 (rules) always runs first as a cheap gate. If it passes and the step is more
critical, the output is escalated to an LLM verifier (L2 for MEDIUM, L3 for HIGH).
This exploits verification asymmetry: most outputs are cleared cheaply, and only
critical or borderline ones pay for a model call.
"""
from __future__ import annotations

from typing import Optional

from ..errors import VerificationResult
from ..pipeline import Criticality
from .base import Verifier, VerifyInput
from .rules import RulesVerifier


class TieredVerifier(Verifier):
    """Composes an L1 rules gate with optional L2/L3 model verifiers."""

    name = "tiered"

    def __init__(
        self,
        *,
        l1: Optional[Verifier] = None,
        l2: Optional[Verifier] = None,
        l3: Optional[Verifier] = None,
    ):
        self.l1 = l1 or RulesVerifier()
        self.l2 = l2  # MEDIUM criticality
        self.l3 = l3 or l2  # HIGH criticality (falls back to l2 if not given)

    async def verify(self, inp: VerifyInput) -> VerificationResult:
        # L1 gate — fail fast on structural problems.
        l1_result = await self.l1.verify(inp)
        if not l1_result.passed:
            return l1_result

        # LOW criticality stops at L1.
        if inp.step.criticality is Criticality.LOW:
            return l1_result

        # Escalate to the appropriate model tier.
        model = self.l3 if inp.step.criticality is Criticality.HIGH else self.l2
        if model is None:
            # No model verifier configured — L1 pass is the final word.
            return l1_result
        return await model.verify(inp)
