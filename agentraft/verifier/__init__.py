"""Verifiers for the AgentRaft protocol."""
from .base import Verifier, VerifyInput
from .mock import MockVerifier
from .rules import RulesVerifier
from .tiered import TieredVerifier

__all__ = [
    "Verifier",
    "VerifyInput",
    "RulesVerifier",
    "MockVerifier",
    "TieredVerifier",
    "LLMVerifier",
]


def __getattr__(name: str):
    # Lazy import so the package never hard-requires openai/anthropic.
    if name == "LLMVerifier":
        from .llm import LLMVerifier

        return LLMVerifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
