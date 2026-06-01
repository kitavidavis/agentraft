"""Verifier interface.

A verifier inspects a single step output in the context of the task goal, the
step's own goal, and the prior verified outputs, and returns a
:class:`~agentraft.errors.VerificationResult`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..errors import VerificationResult
from ..pipeline import Step, Task


@dataclass
class VerifyInput:
    """Bundle of everything a verifier sees about one step output."""

    task: Task
    step: Step
    output: Any
    history: list[tuple[str, Any]]  # (step_name, output) for prior verified steps
    attempt: int = 0


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    async def verify(self, inp: VerifyInput) -> VerificationResult:
        """Judge a step output and return a verdict."""
