"""Scripted verifier for tests and offline demos (no API keys needed).

Lets you pre-program verdicts per step/attempt so you can reproduce a specific
scenario deterministically — e.g. the website's "step 3 drifts, then recovers on
retry" animation.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..errors import ErrorType, VerificationResult
from .base import Verifier, VerifyInput


class MockVerifier(Verifier):
    """Deterministic verifier driven by a script or a callback.

    Args:
        script: maps ``step_name -> list of ErrorType per attempt``. ``ErrorType.NONE``
                means pass. The list is indexed by attempt number; once exhausted the
                last entry repeats.
        default: verdict for steps not present in the script (defaults to pass).
        fn:      optional callable ``(VerifyInput) -> ErrorType`` taking full control.
    """

    name = "mock"

    def __init__(
        self,
        script: Optional[dict[str, list[ErrorType]]] = None,
        *,
        default: ErrorType = ErrorType.NONE,
        fn: Optional[Callable[[VerifyInput], ErrorType]] = None,
    ):
        self.script = script or {}
        self.default = default
        self.fn = fn

    async def verify(self, inp: VerifyInput) -> VerificationResult:
        if self.fn is not None:
            verdict = self.fn(inp)
        elif inp.step.name in self.script:
            attempts = self.script[inp.step.name]
            idx = min(inp.attempt, len(attempts) - 1)
            verdict = attempts[idx]
        else:
            verdict = self.default

        if verdict is ErrorType.NONE:
            return VerificationResult.ok(verifier=self.name, reasoning="mock pass")
        return VerificationResult.fail(
            verdict, reasoning=f"mock {verdict.value}", verifier=self.name
        )
