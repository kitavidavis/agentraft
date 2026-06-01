import pytest

from agentraft import (
    Criticality,
    ErrorType,
    EventType,
    MockVerifier,
    Pipeline,
    Step,
    StepContext,
    Task,
    wrap,
)


async def _const(value):
    async def fn(ctx: StepContext):
        return value
    return fn


@pytest.mark.asyncio
async def test_happy_path_commits_all_steps(simple_pipeline):
    coord = wrap(simple_pipeline, verifier=MockVerifier())
    result = await coord.run(Task(goal="run"))

    assert result.success
    assert result.verified_count == 3
    assert result.rollbacks == 0
    assert result.reliability == 1.0
    assert result.output == "alpha-beta-gamma"


@pytest.mark.asyncio
async def test_drift_then_recover():
    calls = {"n": 0}

    async def draft(ctx: StepContext):
        calls["n"] += 1
        return "bad" if ctx.attempt == 0 else "good"

    pipe = Pipeline([Step("draft", draft, criticality=Criticality.HIGH)])
    verifier = MockVerifier({"draft": [ErrorType.GOAL_DRIFT, ErrorType.NONE]})
    coord = wrap(pipe, verifier=verifier)

    result = await coord.run(Task(goal="write"))

    assert result.success
    assert result.rollbacks == 1
    assert calls["n"] == 2  # ran twice: drift, then recover
    assert result.output == "good"


@pytest.mark.asyncio
async def test_max_retries_exceeded_fails_run():
    async def always_bad(ctx: StepContext):
        return "bad"

    pipe = Pipeline([Step("s", always_bad)])
    verifier = MockVerifier({"s": [ErrorType.HALLUCINATION]})  # always fails
    coord = wrap(pipe, verifier=verifier, max_retries=2)

    result = await coord.run(Task(goal="x"))

    assert not result.success
    assert "HALLUCINATION" in result.error
    assert result.steps[-1].status == "failed"
    assert result.steps[-1].attempts == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_correction_hint_is_passed_on_retry():
    seen_hints = []

    async def s(ctx: StepContext):
        seen_hints.append(ctx.hint)
        return "v"

    pipe = Pipeline([Step("s", s)])
    verifier = MockVerifier({"s": [ErrorType.INCOMPLETE, ErrorType.NONE]})
    coord = wrap(pipe, verifier=verifier)

    await coord.run(Task(goal="x"))

    assert seen_hints[0] is None            # first attempt has no hint
    assert seen_hints[1] is not None        # retry receives a typed hint
    assert "missing" in seen_hints[1].lower()


@pytest.mark.asyncio
async def test_events_emitted_in_order():
    events = []

    async def s(ctx: StepContext):
        return "v"

    pipe = Pipeline([Step("s", s)])
    coord = wrap(pipe, verifier=MockVerifier(), on_event=lambda e: events.append(e.type))

    await coord.run(Task(goal="x"))

    assert events[0] == EventType.RUN_START
    assert EventType.STEP_COMMITTED in events
    assert events[-1] == EventType.RUN_SUCCESS


@pytest.mark.asyncio
async def test_run_or_raise_raises_on_failure():
    from agentraft import MaxRetriesExceeded

    async def bad(ctx: StepContext):
        return "bad"

    pipe = Pipeline([Step("s", bad)])
    coord = wrap(pipe, verifier=MockVerifier({"s": [ErrorType.GOAL_DRIFT]}), max_retries=1)

    with pytest.raises(MaxRetriesExceeded):
        await coord.run_or_raise(Task(goal="x"))
