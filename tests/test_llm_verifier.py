"""Offline tests for the multi-provider LLM verifier.

These never touch the network — they inject a fake ``complete_fn`` so we can test
verdict parsing, the confidence threshold, and provider routing deterministically.
"""
import pytest

from agentraft import Criticality, ErrorType, Step, Task
from agentraft.verifier import VerifyInput
from agentraft.verifier.llm import DEFAULT_MODELS, LLMVerifier, _parse_verdict


def _inp(output="some output"):
    return VerifyInput(
        task=Task(goal="write a memo"),
        step=Step("draft", lambda ctx: output, goal="write an on-topic memo",
                  criticality=Criticality.HIGH),
        output=output,
        history=[("research", "sources gathered")],
    )


def _fake(verdict_json: str):
    def fn(system, user):
        return verdict_json
    return fn


# ── verdict parsing ──────────────────────────────────────────────────────────
def test_parse_pass():
    r = _parse_verdict('{"verdict":"PASS","confidence":0.9,"reasoning":"good"}', "v")
    assert r.passed and r.error_type is ErrorType.NONE


def test_parse_failure_class_and_hint():
    r = _parse_verdict('{"verdict":"GOAL_DRIFT","confidence":0.8,"hint":"refocus"}', "v")
    assert not r.passed
    assert r.error_type is ErrorType.GOAL_DRIFT
    assert r.hint == "refocus"


def test_parse_tolerates_surrounding_prose():
    raw = 'Sure! Here is my verdict:\n{"verdict":"PASS","confidence":0.95}\nThanks.'
    assert _parse_verdict(raw, "v").passed


def test_parse_unparseable_is_soft_incomplete():
    r = _parse_verdict("no json here", "v")
    assert not r.passed
    assert r.error_type is ErrorType.INCOMPLETE


# ── verify() end-to-end (offline) ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_verify_pass_offline():
    v = LLMVerifier(provider="bedrock", complete_fn=_fake('{"verdict":"PASS","confidence":0.92}'))
    r = await v.verify(_inp())
    assert r.passed


@pytest.mark.asyncio
async def test_verify_fail_offline():
    v = LLMVerifier(provider="openai",
                    complete_fn=_fake('{"verdict":"HALLUCINATION","confidence":0.8,"hint":"stick to context"}'))
    r = await v.verify(_inp())
    assert not r.passed
    assert r.error_type is ErrorType.HALLUCINATION


@pytest.mark.asyncio
async def test_low_confidence_pass_becomes_soft_fail():
    v = LLMVerifier(provider="anthropic", pass_threshold=0.7,
                    complete_fn=_fake('{"verdict":"PASS","confidence":0.4}'))
    r = await v.verify(_inp())
    assert not r.passed
    assert r.error_type is ErrorType.INCOMPLETE


# ── provider routing / config ────────────────────────────────────────────────
def test_bedrock_default_model():
    v = LLMVerifier.bedrock()
    assert v.provider == "bedrock"
    assert v.model == DEFAULT_MODELS["bedrock"]
    assert v.name.startswith("llm-bedrock:")


def test_bedrock_custom_model_any_family():
    # The Converse API supports many model families by id alone.
    for mid in [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "mistral.mistral-large-2407-v1:0",
        "amazon.nova-lite-v1:0",
        "cohere.command-r-plus-v1:0",
    ]:
        v = LLMVerifier.bedrock(model=mid, region="us-west-2")
        assert v.model == mid
        assert v.region == "us-west-2"


def test_provider_convenience_constructors():
    assert LLMVerifier.openai().provider == "openai"
    assert LLMVerifier.anthropic().provider == "anthropic"
    assert LLMVerifier.gemini().provider == "gemini"


def test_explicit_provider_skips_detection(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    # No env configured, but explicit provider must still construct fine.
    v = LLMVerifier(provider="bedrock")
    assert v.provider == "bedrock"


def test_forced_provider_via_env(monkeypatch):
    monkeypatch.setenv("AGENTRAFT_VERIFIER_PROVIDER", "gemini")
    assert LLMVerifier._detect_provider() == "gemini"
