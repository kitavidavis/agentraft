"""L1 rules verifier — cheap, deterministic, no model calls.

This is the first gate in the tiered pipeline. It catches the obvious failures
(empty output, error strings, truncation, missing required keywords) for ~free,
so the expensive LLM verifier only runs on outputs that pass these checks.
"""
from __future__ import annotations

import re
from typing import Optional

from ..errors import ErrorType, VerificationResult
from .base import Verifier, VerifyInput

_ERROR_SIGNATURES = (
    "traceback (most recent call last)",
    "i'm sorry, i can't",
    "i cannot help with that",
    "as an ai language model",
    "[error]",
    "null",
    "undefined",
)


class RulesVerifier(Verifier):
    """Structural sanity checks. Fast L1 gate."""

    name = "rules-l1"

    def __init__(
        self,
        *,
        min_length: int = 1,
        max_length: Optional[int] = None,
        required_keywords: Optional[list[str]] = None,
        forbid_error_signatures: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.required_keywords = [k.lower() for k in (required_keywords or [])]
        self.forbid_error_signatures = forbid_error_signatures

    async def verify(self, inp: VerifyInput) -> VerificationResult:
        text = "" if inp.output is None else str(inp.output)
        stripped = text.strip()

        # Empty / missing output -> INCOMPLETE
        if len(stripped) < self.min_length:
            return VerificationResult.fail(
                ErrorType.INCOMPLETE,
                reasoning="Output is empty or below the minimum length.",
                verifier=self.name,
            )

        # Known refusal / error signatures -> INCOMPLETE
        if self.forbid_error_signatures:
            low = stripped.lower()
            for sig in _ERROR_SIGNATURES:
                if sig in low:
                    return VerificationResult.fail(
                        ErrorType.INCOMPLETE,
                        reasoning=f"Output contains an error/refusal signature: {sig!r}.",
                        verifier=self.name,
                    )

        # Over-long output -> SCOPE_CREEP (a soft signal it did too much)
        if self.max_length is not None and len(stripped) > self.max_length:
            return VerificationResult.fail(
                ErrorType.SCOPE_CREEP,
                reasoning=f"Output exceeds max_length ({len(stripped)} > {self.max_length}).",
                verifier=self.name,
            )

        # Missing required keywords -> INCOMPLETE
        if self.required_keywords:
            low = stripped.lower()
            missing = [k for k in self.required_keywords if k not in low]
            if missing:
                return VerificationResult.fail(
                    ErrorType.INCOMPLETE,
                    reasoning=f"Missing required keyword(s): {', '.join(missing)}.",
                    verifier=self.name,
                )

        # Dangling sentence (truncation) -> INCOMPLETE, low confidence
        if re.search(r"[A-Za-z],?\s*$", stripped) and not re.search(r"[.!?\")\]]\s*$", stripped):
            return VerificationResult.fail(
                ErrorType.INCOMPLETE,
                reasoning="Output appears to be cut off mid-sentence.",
                confidence=0.6,
                verifier=self.name,
            )

        return VerificationResult.ok(verifier=self.name, reasoning="Passed L1 structural checks.")
