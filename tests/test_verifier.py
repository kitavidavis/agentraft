import pytest

from agentraft import Criticality, ErrorType, RulesVerifier, Step, Task
from agentraft.verifier import MockVerifier, TieredVerifier, VerifyInput


def _inp(output, *, goal=None, criticality=Criticality.MEDIUM, attempt=0):
    return VerifyInput(
        task=Task(goal="test task"),
        step=Step("s", lambda ctx: output, goal=goal, criticality=criticality),
        output=output,
        history=[],
        attempt=attempt,
    )


@pytest.mark.asyncio
async def test_rules_rejects_empty():
    r = await RulesVerifier().verify(_inp(""))
    assert not r.passed
    assert r.error_type is ErrorType.INCOMPLETE


@pytest.mark.asyncio
async def test_rules_rejects_error_signature():
    r = await RulesVerifier().verify(_inp("Traceback (most recent call last): boom"))
    assert not r.passed
    assert r.error_type is ErrorType.INCOMPLETE


@pytest.mark.asyncio
async def test_rules_required_keywords():
    v = RulesVerifier(required_keywords=["summary"])
    assert (await v.verify(_inp("here is the summary."))).passed
    assert not (await v.verify(_inp("here is the report."))).passed


@pytest.mark.asyncio
async def test_rules_max_length_scope_creep():
    v = RulesVerifier(max_length=10)
    r = await v.verify(_inp("this output is definitely way too long."))
    assert not r.passed
    assert r.error_type is ErrorType.SCOPE_CREEP


@pytest.mark.asyncio
async def test_rules_passes_clean_output():
    assert (await RulesVerifier().verify(_inp("A clean, complete sentence."))).passed


@pytest.mark.asyncio
async def test_tiered_low_criticality_stops_at_l1():
    # L2 would reject everything; LOW criticality must never reach it.
    always_fail = MockVerifier(default=ErrorType.GOAL_DRIFT)
    tv = TieredVerifier(l1=RulesVerifier(), l2=always_fail)
    r = await tv.verify(_inp("clean output.", criticality=Criticality.LOW))
    assert r.passed
    assert r.verifier == "rules-l1"


@pytest.mark.asyncio
async def test_tiered_escalates_medium_to_l2():
    always_fail = MockVerifier(default=ErrorType.GOAL_DRIFT)
    tv = TieredVerifier(l1=RulesVerifier(), l2=always_fail)
    r = await tv.verify(_inp("clean output.", criticality=Criticality.MEDIUM))
    assert not r.passed
    assert r.error_type is ErrorType.GOAL_DRIFT


@pytest.mark.asyncio
async def test_tiered_l1_gate_blocks_before_l2():
    # Empty output should be caught by L1 and never escalate.
    never_called = MockVerifier(default=ErrorType.NONE)
    tv = TieredVerifier(l1=RulesVerifier(), l2=never_called)
    r = await tv.verify(_inp("", criticality=Criticality.HIGH))
    assert not r.passed
    assert r.verifier == "rules-l1"


@pytest.mark.asyncio
async def test_mock_script_per_attempt():
    v = MockVerifier({"s": [ErrorType.GOAL_DRIFT, ErrorType.NONE]})
    assert not (await v.verify(_inp("x", attempt=0))).passed
    assert (await v.verify(_inp("x", attempt=1))).passed
