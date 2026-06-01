"""L2/L3 LLM verifier — the ML heart of AgentRaft.

Exploits *verification asymmetry*: judging whether a step output satisfies its
goal is far cheaper than generating it, so a small model (or a cheap frontier
tier) can verify the work of a much larger agent.

Supported providers
--------------------
- ``bedrock``   — Amazon Bedrock via the unified **Converse API**. One code path
                  covers every chat model on Bedrock: Anthropic Claude, Meta
                  Llama, Mistral, Amazon Nova/Titan, Cohere, AI21. This is where
                  most enterprise agents run, so it's a first-class provider.
- ``openai``    — OpenAI GPT models.
- ``anthropic`` — Anthropic Claude (direct API).
- ``gemini``    — Google Gemini.

All providers return a structured JSON verdict that is parsed into a
:class:`VerificationResult`. Swap providers without touching pipeline code.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Callable, Literal, Optional

from ..errors import ErrorType, VerificationResult, VerifierUnavailable
from .base import Verifier, VerifyInput

Provider = Literal["bedrock", "openai", "anthropic", "gemini"]

_VALID_CLASSES = {e.value for e in ErrorType}

#: Sensible default model per provider. Override with ``model=`` or env var.
DEFAULT_MODELS: dict[str, str] = {
    "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",  # broadly on-demand
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-1.5-flash",
}

SYSTEM_PROMPT = """You are AgentRaft's step verifier. You judge whether a single step's \
output satisfies its goal within a multi-step AI pipeline.

You DO NOT do the task yourself. You ONLY evaluate the given output and assign exactly \
one verdict from this taxonomy:

- PASS          : output correctly and completely satisfies the step goal.
- GOAL_DRIFT    : output diverges from the task/step objective.
- CONTRADICTION : output contradicts a previously verified step.
- HALLUCINATION : output asserts facts not supported by the task inputs or prior outputs.
- INCOMPLETE    : output is missing required elements.
- SCOPE_CREEP   : output introduces actions/content outside this step's scope.

Respond with STRICT JSON only, no prose:
{"verdict": "PASS|GOAL_DRIFT|CONTRADICTION|HALLUCINATION|INCOMPLETE|SCOPE_CREEP",
 "confidence": 0.0-1.0,
 "reasoning": "one concise sentence",
 "hint": "if not PASS, a concrete instruction to fix it on retry; else empty"}"""


def _build_user_prompt(inp: VerifyInput) -> str:
    history = "\n".join(
        f"  - {name}: {str(out)[:500]}" for name, out in inp.history
    ) or "  (none yet)"
    return (
        f"TASK GOAL:\n{inp.task.goal}\n\n"
        f"TASK INPUTS:\n{json.dumps(inp.task.inputs, default=str)[:1000]}\n\n"
        f"PRIOR VERIFIED STEP OUTPUTS:\n{history}\n\n"
        f"CURRENT STEP: {inp.step.name}\n"
        f"STEP GOAL: {inp.step.goal or '(not specified — infer from task goal)'}\n\n"
        f"STEP OUTPUT TO VERIFY:\n{str(inp.output)[:4000]}\n\n"
        f"Return your JSON verdict."
    )


def _parse_verdict(raw: str, verifier_name: str) -> VerificationResult:
    """Parse the model's JSON verdict robustly (tolerates surrounding prose)."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return VerificationResult.fail(
            ErrorType.INCOMPLETE,
            reasoning="Verifier returned unparseable output.",
            confidence=0.3,
            verifier=verifier_name,
        )
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return VerificationResult.fail(
            ErrorType.INCOMPLETE,
            reasoning="Verifier returned malformed JSON.",
            confidence=0.3,
            verifier=verifier_name,
        )
    verdict = str(data.get("verdict", "")).upper().strip()
    confidence = float(data.get("confidence", 0.8))
    reasoning = str(data.get("reasoning", ""))[:300]
    hint = str(data.get("hint", "")).strip() or None

    if verdict == "PASS":
        return VerificationResult.ok(confidence=confidence, reasoning=reasoning, verifier=verifier_name)
    if verdict in _VALID_CLASSES and verdict != "NONE":
        return VerificationResult.fail(
            ErrorType(verdict), hint=hint, confidence=confidence,
            reasoning=reasoning, verifier=verifier_name,
        )
    return VerificationResult.fail(
        ErrorType.GOAL_DRIFT,
        reasoning=f"Unrecognized verdict label: {verdict!r}.",
        confidence=0.4,
        verifier=verifier_name,
    )


class LLMVerifier(Verifier):
    """Verifier backed by a hosted LLM. Provider-agnostic.

    Convenience constructors::

        LLMVerifier.bedrock(model="anthropic.claude-3-5-sonnet-20241022-v2:0", region="us-east-1")
        LLMVerifier.bedrock(model="meta.llama3-1-70b-instruct-v1:0")   # any Bedrock model
        LLMVerifier.openai(model="gpt-4o-mini")
        LLMVerifier.anthropic(model="claude-3-5-haiku-latest")
        LLMVerifier.gemini(model="gemini-1.5-flash")
    """

    def __init__(
        self,
        *,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
        pass_threshold: float = 0.5,
        complete_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        self.provider: Provider = provider or self._detect_provider()
        self.model = model or os.getenv("AGENTRAFT_VERIFIER_MODEL") or DEFAULT_MODELS[self.provider]
        self.api_key = api_key
        self.region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pass_threshold = pass_threshold
        self.name = f"llm-{self.provider}:{self.model}"
        self._client = None
        # Injectable completion fn (used by tests / custom transports). Signature:
        #   (system: str, user: str) -> str | Awaitable[str]
        self._complete_fn = complete_fn

    # ── Convenience constructors ────────────────────────────────────────────
    @classmethod
    def bedrock(cls, *, model: Optional[str] = None, region: Optional[str] = None, **kw) -> "LLMVerifier":
        return cls(provider="bedrock", model=model, region=region, **kw)

    @classmethod
    def openai(cls, *, model: Optional[str] = None, api_key: Optional[str] = None, **kw) -> "LLMVerifier":
        return cls(provider="openai", model=model, api_key=api_key, **kw)

    @classmethod
    def anthropic(cls, *, model: Optional[str] = None, api_key: Optional[str] = None, **kw) -> "LLMVerifier":
        return cls(provider="anthropic", model=model, api_key=api_key, **kw)

    @classmethod
    def gemini(cls, *, model: Optional[str] = None, api_key: Optional[str] = None, **kw) -> "LLMVerifier":
        return cls(provider="gemini", model=model, api_key=api_key, **kw)

    # ── Provider detection ───────────────────────────────────────────────────
    @staticmethod
    def _detect_provider() -> Provider:
        forced = os.getenv("AGENTRAFT_VERIFIER_PROVIDER")
        if forced:
            forced = forced.lower().strip()
            if forced not in DEFAULT_MODELS:
                raise VerifierUnavailable(f"Unknown AGENTRAFT_VERIFIER_PROVIDER={forced!r}")
            return forced  # type: ignore[return-value]
        if os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AGENTRAFT_USE_BEDROCK"):
            return "bedrock"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "gemini"
        # AWS credentials present (profile / role / keys) -> assume Bedrock.
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
            return "bedrock"
        raise VerifierUnavailable(
            "No LLM provider configured. Set one of: AWS credentials (Bedrock), "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY — or pass "
            "provider=/api_key= explicitly."
        )

    # ── Completion dispatch ───────────────────────────────────────────────────
    async def _complete(self, system: str, user: str) -> str:
        if self._complete_fn is not None:
            res = self._complete_fn(system, user)
            return await res if asyncio.iscoroutine(res) else res
        if self.provider == "bedrock":
            return await self._complete_bedrock(system, user)
        if self.provider == "openai":
            return await self._complete_openai(system, user)
        if self.provider == "anthropic":
            return await self._complete_anthropic(system, user)
        if self.provider == "gemini":
            return await self._complete_gemini(system, user)
        raise VerifierUnavailable(f"Unknown provider {self.provider!r}")  # pragma: no cover

    async def _complete_bedrock(self, system: str, user: str) -> str:
        """Amazon Bedrock via the unified Converse API (works for all chat models)."""
        if self._client is None:
            try:
                import boto3  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise VerifierUnavailable("pip install agentraft[bedrock]") from e
            self._client = boto3.client("bedrock-runtime", region_name=self.region)

        def _call() -> str:
            resp = self._client.converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens},
            )
            return resp["output"]["message"]["content"][0]["text"]

        # boto3 is sync — run it off the event loop.
        return await asyncio.to_thread(_call)

    async def _complete_openai(self, system: str, user: str) -> str:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise VerifierUnavailable("pip install agentraft[openai]") from e
            self._client = AsyncOpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        resp = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    async def _complete_anthropic(self, system: str, user: str) -> str:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:  # pragma: no cover
                raise VerifierUnavailable("pip install agentraft[anthropic]") from e
            self._client = AsyncAnthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        resp = await self._client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text if resp.content else ""

    async def _complete_gemini(self, system: str, user: str) -> str:
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise VerifierUnavailable("pip install agentraft[google]") from e
            self._client = genai.Client(api_key=self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        resp = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
            },
        )
        return resp.text or ""

    # ── Verify ────────────────────────────────────────────────────────────────
    async def verify(self, inp: VerifyInput) -> VerificationResult:
        raw = await self._complete(SYSTEM_PROMPT, _build_user_prompt(inp))
        result = _parse_verdict(raw, self.name)
        # Treat low-confidence passes as soft failures (escalation policy).
        if result.passed and result.confidence < self.pass_threshold:
            return VerificationResult.fail(
                ErrorType.INCOMPLETE,
                reasoning=f"Pass confidence {result.confidence:.2f} below threshold "
                          f"{self.pass_threshold:.2f}.",
                confidence=result.confidence,
                verifier=self.name,
            )
        return result
