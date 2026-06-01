"""Error taxonomy and exceptions for the AgentRaft protocol.

The taxonomy is the heart of step-level verification: instead of a binary
pass/fail, the verifier classifies *why* a step failed, and that class maps to a
typed correction hint that is injected into the next retry attempt.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorType(str, Enum):
    """The five failure classes a verifier can assign to a step output."""

    NONE = "NONE"
    GOAL_DRIFT = "GOAL_DRIFT"
    CONTRADICTION = "CONTRADICTION"
    HALLUCINATION = "HALLUCINATION"
    INCOMPLETE = "INCOMPLETE"
    SCOPE_CREEP = "SCOPE_CREEP"

    @property
    def is_failure(self) -> bool:
        return self is not ErrorType.NONE


#: Default correction hint injected into the retry when a class is detected.
#: A custom verifier may override these with context-specific guidance.
CORRECTION_HINTS: dict[ErrorType, str] = {
    ErrorType.GOAL_DRIFT: (
        "The previous attempt drifted from the task objective. Refocus strictly on the "
        "stated goal and remove any tangential or unrelated content."
    ),
    ErrorType.CONTRADICTION: (
        "The previous attempt contradicted an earlier verified step. Re-read the prior "
        "outputs and produce something consistent with them."
    ),
    ErrorType.HALLUCINATION: (
        "The previous attempt asserted facts not supported by the provided context. Only "
        "use information present in the task inputs and prior verified outputs."
    ),
    ErrorType.INCOMPLETE: (
        "The previous attempt was missing required elements. Re-read the step goal and "
        "ensure every required part is present in the output."
    ),
    ErrorType.SCOPE_CREEP: (
        "The previous attempt introduced actions or content outside this step's scope. "
        "Do only what this step requires — nothing more."
    ),
}


def hint_for(error_type: ErrorType) -> Optional[str]:
    """Return the default correction hint for an error class (None if no failure)."""
    return CORRECTION_HINTS.get(error_type)


@dataclass
class VerificationResult:
    """The verdict a verifier returns for a single step output."""

    passed: bool
    error_type: ErrorType = ErrorType.NONE
    confidence: float = 1.0
    hint: Optional[str] = None
    reasoning: str = ""
    verifier: str = ""  # name of the verifier that produced this verdict

    @classmethod
    def ok(cls, *, confidence: float = 1.0, verifier: str = "", reasoning: str = "") -> "VerificationResult":
        return cls(passed=True, error_type=ErrorType.NONE, confidence=confidence,
                   verifier=verifier, reasoning=reasoning)

    @classmethod
    def fail(
        cls,
        error_type: ErrorType,
        *,
        hint: Optional[str] = None,
        confidence: float = 1.0,
        reasoning: str = "",
        verifier: str = "",
    ) -> "VerificationResult":
        if error_type is ErrorType.NONE:
            raise ValueError("VerificationResult.fail requires a real ErrorType, not NONE")
        return cls(
            passed=False,
            error_type=error_type,
            confidence=confidence,
            hint=hint or hint_for(error_type),
            reasoning=reasoning,
            verifier=verifier,
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "error_type": self.error_type.value,
            "confidence": round(self.confidence, 4),
            "hint": self.hint,
            "reasoning": self.reasoning,
            "verifier": self.verifier,
        }


# ── Exceptions ──────────────────────────────────────────────────────────────
class AgentRaftError(Exception):
    """Base class for all AgentRaft runtime errors."""


class CircuitBreakerOpen(AgentRaftError):
    """Raised when the circuit breaker is open and refuses further attempts."""


class MaxRetriesExceeded(AgentRaftError):
    """Raised when a step fails verification more times than allowed."""

    def __init__(self, step_name: str, attempts: int, last: Optional[VerificationResult] = None):
        self.step_name = step_name
        self.attempts = attempts
        self.last = last
        detail = f" (last: {last.error_type.value})" if last else ""
        super().__init__(
            f"Step '{step_name}' failed verification after {attempts} attempt(s){detail}"
        )


class VerifierUnavailable(AgentRaftError):
    """Raised when a configured verifier backend cannot be reached or used."""
