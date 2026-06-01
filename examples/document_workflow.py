"""Runnable demo: the enterprise document workflow from the AgentRaft landing page.

A 5-step pipeline (research -> analyse -> draft -> review -> publish). Step 3
intentionally drifts on its first attempt; AgentRaft catches it, rolls back, and
retries with a correction hint, then completes.

Run it with zero setup (uses the scripted MockVerifier):

    python -m examples.document_workflow

Or against a real LLM verifier:

    export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY
    python -m examples.document_workflow --live
"""
from __future__ import annotations

import asyncio
import sys

# Use UTF-8 for the icons/box-drawing below, even on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - older/odd stdio
    pass

from agentraft import Criticality, ErrorType, Pipeline, Step, StepContext, Task, wrap  # noqa: E402
from agentraft.events import Event, EventType  # noqa: E402
from agentraft.verifier import MockVerifier  # noqa: E402

# ── Pipeline steps ──────────────────────────────────────────────────────────
# Each step is a normal async function. `drift_once` simulates a non-deterministic
# agent that produces a bad output the first time and a good one after a hint.

_draft_calls = {"n": 0}


async def research(ctx: StepContext) -> str:
    return f"Collected 7 sources on: {ctx.task.goal}"


async def analyse(ctx: StepContext) -> str:
    return "Key themes: reliability gap, verification asymmetry, enterprise adoption."


async def draft(ctx: StepContext) -> str:
    _draft_calls["n"] += 1
    if ctx.is_retry or _draft_calls["n"] > 1:
        # After the correction hint, produce an on-goal draft.
        return (
            "Draft board memo: AgentRaft closes the agentic-AI reliability gap via "
            "step-level verification and rollback. Recommends a Q3 design-partner pilot."
        )
    # First attempt drifts off-topic.
    return "Here are ten fun marketing taglines we could tweet about AI agents!"


async def review(ctx: StepContext) -> str:
    return "Reviewed: tone consistent, claims supported by analysis. Approved."


async def publish(ctx: StepContext) -> str:
    return "Published board memo to the shared drive and notified stakeholders."


def build_pipeline() -> Pipeline:
    return Pipeline([
        Step("research_agent", research, goal="Gather relevant sources for the memo"),
        Step("analysis_agent", analyse, goal="Extract the key themes from the sources"),
        Step("draft_agent",   draft,   goal="Write an on-topic board memo about the task goal",
             criticality=Criticality.HIGH),
        Step("review_agent",  review,  goal="Check the draft for accuracy and tone"),
        Step("publish_agent", publish, goal="Publish the approved memo"),
    ])


def print_event(e: Event) -> None:
    icons = {
        EventType.STEP_START: "▶",  EventType.STEP_VERIFYING: "🔍",
        EventType.STEP_COMMITTED: "✓", EventType.STEP_FAILED: "✗",
        EventType.STEP_ROLLBACK: "↺", EventType.STEP_RETRY: "⟳",
        EventType.RUN_SUCCESS: "🎉", EventType.RUN_FAILED: "💥",
        EventType.BREAKER_OPEN: "⚡",
    }
    icon = icons.get(e.type, "·")
    if e.type in (EventType.STEP_COMMITTED, EventType.STEP_FAILED):
        v = e.verification
        verdict = "COMMITTED" if e.verification and e.verification.passed else (
            v.error_type.value if v else "FAILED")
        print(f"  {icon} {e.step_name:<16} {verdict}")
    elif e.type == EventType.STEP_RETRY:
        print(f"  {icon} {e.step_name:<16} retry with hint")
    elif e.type in (EventType.RUN_SUCCESS, EventType.RUN_FAILED):
        print(f"\n{icon} {e.type.value}")


async def main(live: bool = False) -> None:
    pipeline = build_pipeline()

    if live:
        # Real LLM verifier (needs OPENAI_API_KEY or ANTHROPIC_API_KEY).
        coordinator = wrap(pipeline, on_event=print_event)
        print("Running with LIVE LLM verifier…\n")
    else:
        # Scripted verifier: draft_agent drifts on attempt 0, passes on attempt 1.
        verifier = MockVerifier({
            "draft_agent": [ErrorType.GOAL_DRIFT, ErrorType.NONE],
        })
        coordinator = wrap(pipeline, verifier=verifier, on_event=print_event)
        print("Running with scripted MockVerifier (no API key needed)…\n")

    task = Task(goal="Write the Q3 board memo on AgentRaft's reliability thesis")
    result = await coordinator.run(task)

    print("\n── Result " + "─" * 40)
    for k, v in result.summary().items():
        print(f"  {k:<12}: {v}")
    print("\nFinal output:\n  " + str(result.output))


if __name__ == "__main__":
    asyncio.run(main(live="--live" in sys.argv))
