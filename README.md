# AgentRaft

**Distributed reliability infrastructure for agentic AI.**
Raft-inspired, step-level consensus for multi-step agent pipelines — verify every step before it commits, and roll back to the last good checkpoint on failure.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The problem

Agent reliability compounds in the wrong direction. At **95% per-step reliability**, a 20-step pipeline succeeds only **~36% of the time** (`0.95²⁰ ≈ 0.36`). Errors propagate confidently and silently, which is why enterprises can't ship agentic AI for critical workflows.

## The idea

AgentRaft borrows from the [Raft consensus algorithm](https://raft.github.io/): nothing is committed to the log without agreement. Here, a cheap **verifier** stands in for the quorum — it judges each step's output *before* it's committed to a checkpoint store. Verification is far cheaper than generation (*verification asymmetry*), so a small 3–7B model can guard the work of a much larger agent at **10–100× lower cost** than re-running the pipeline.

```
run step → verify → ✓ commit checkpoint
                  → ✗ classify error → rollback → retry with typed hint
                                                 → circuit breaker if it cascades
```

## Install

```bash
pip install agentraft                # core (zero deps, rules-only verifier)
pip install "agentraft[bedrock]"     # + Amazon Bedrock verifier (Converse API)
pip install "agentraft[openai]"      # + OpenAI verifier
pip install "agentraft[anthropic]"   # + Anthropic verifier
pip install "agentraft[google]"      # + Google Gemini verifier
pip install "agentraft[redis]"       # + durable Redis checkpoint store
pip install "agentraft[all]"         # everything
```

## Providers — built for where agents actually run

Most enterprise agents run on **Amazon Bedrock**, so it's a first-class provider. AgentRaft uses the unified **Bedrock Converse API**, which means a *single* verifier code path supports every chat model on Bedrock — just change the model id:

```python
from agentraft import wrap
from agentraft.verifier import LLMVerifier, TieredVerifier, RulesVerifier

# Amazon Bedrock — Claude, Llama, Mistral, Amazon Nova, Cohere, AI21
verifier = TieredVerifier(l1=RulesVerifier(), l2=LLMVerifier.bedrock(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",   # or meta.llama3-1-70b-instruct-v1:0, amazon.nova-lite-v1:0, …
    region="us-east-1",
))
coordinator = wrap(pipeline, verifier=verifier)
```

| Provider | Constructor | Models |
|---|---|---|
| **Amazon Bedrock** | `LLMVerifier.bedrock(model=…)` | Claude · Llama · Mistral · Amazon Nova/Titan · Cohere · AI21 |
| OpenAI | `LLMVerifier.openai(model=…)` | GPT-4o, GPT-4o-mini, … |
| Anthropic | `LLMVerifier.anthropic(model=…)` | Claude (direct API) |
| Google | `LLMVerifier.gemini(model=…)` | Gemini 1.5/2.x |

`wrap()` **auto-detects** the provider from the environment: AWS credentials → Bedrock, else `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`. Force one with `AGENTRAFT_VERIFIER_PROVIDER=bedrock` and `AGENTRAFT_VERIFIER_MODEL=…`.

## Quickstart

```python
import asyncio
from agentraft import wrap, Pipeline, Step, Task, Criticality

async def research(ctx):  return f"Sources on: {ctx.task.goal}"
async def draft(ctx):     return "Board memo draft …"
async def review(ctx):    return "Reviewed and approved."

pipeline = Pipeline([
    Step("research", research, goal="Gather relevant sources"),
    Step("draft",    draft,    goal="Write an on-topic memo", criticality=Criticality.HIGH),
    Step("review",   review,   goal="Check accuracy and tone"),
])

async def main():
    result = await wrap(pipeline).run(Task(goal="Write the Q3 board memo"))
    print(result.summary())   # {'success': True, 'verified': '3/3', 'rollbacks': 0, ...}
    print(result.output)

asyncio.run(main())
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and `wrap()` automatically uses a tiered verifier (L1 rules → LLM). With no key, it runs rules-only.

## Run the demo (no API key needed)

```bash
python -m examples.document_workflow          # scripted: step 3 drifts, then recovers
python -m examples.document_workflow --live   # uses a real LLM verifier
```

```
  ▶ research_agent
  ✓ research_agent     COMMITTED
  ✓ analysis_agent     COMMITTED
  ✗ draft_agent        GOAL_DRIFT
  ↺ draft_agent        rollback → checkpoint_2
  ⟳ draft_agent        retry with hint
  ✓ draft_agent        COMMITTED
  ✓ review_agent       COMMITTED
  ✓ publish_agent      COMMITTED
🎉 run_success   verified 5/5 · rollbacks 1 · reliability 1.0
```

## Architecture

AgentRaft is composed of five replaceable components:

| Component | Role | Default impl |
|---|---|---|
| **Coordinator** | Runs the consensus loop — sequence, verify, commit, rollback, retry | `coordinator.py` |
| **Worker Agent** | Your existing pipeline steps — unchanged | your code |
| **Verifier** | Judges each step against its goal; assigns a typed error class | `RulesVerifier` + `LLMVerifier`, routed by `TieredVerifier` |
| **Checkpoint Store** | Append-only log of verified outputs; rollback target | `InMemoryCheckpointStore` / `RedisCheckpointStore` |
| **Circuit Breaker** | Stops error cascades and runaway cost | `CircuitBreaker` + `RetryPolicy` |

### Error taxonomy

Verification isn't binary. Each failure is classified, and the class maps to a typed correction hint injected into the retry:

| Class | Meaning |
|---|---|
| `GOAL_DRIFT` | Output diverges from the task objective |
| `CONTRADICTION` | Output contradicts a previously verified step |
| `HALLUCINATION` | Asserts facts unsupported by the context |
| `INCOMPLETE` | Required elements are missing |
| `SCOPE_CREEP` | Introduces out-of-scope actions |

### Tiered verification

`TieredVerifier` runs the cheap **L1 rules** gate on every step, then escalates by step criticality:

- `Criticality.LOW` → L1 rules only
- `Criticality.MEDIUM` → L2 (small LLM verifier)
- `Criticality.HIGH` → L3 (large LLM verifier)

Most outputs clear at L1 for free; only critical or borderline ones pay for a model call.

## Live monitoring

Pass an `on_event` hook to stream protocol events (`STEP_COMMITTED`, `STEP_ROLLBACK`, …) into a dashboard, logger, or the live monitor on [agentraft.io](https://agentraft.io):

```python
from agentraft import wrap, EventType

def on_event(e):
    if e.type == EventType.STEP_ROLLBACK:
        print("rolled back", e.step_name)

coordinator = wrap(pipeline, on_event=on_event)
```

## Configuration

```python
from agentraft import wrap, RulesVerifier, TieredVerifier
from agentraft.verifier import LLMVerifier

coordinator = wrap(
    pipeline,
    verifier=TieredVerifier(l1=RulesVerifier(), l2=LLMVerifier(provider="anthropic")),
    max_retries=3,            # per-step retry budget
    failure_threshold=5,      # consecutive failures before the breaker opens
    cooldown_seconds=30,      # breaker cooldown
    rollback_on_failure=True, # revert checkpoints before retrying
)
```

## Development

```bash
pip install -e ".[dev]"
pytest                # run the test suite
ruff check .          # lint
```

## Status

Early alpha. The Python SDK is the reference implementation of the protocol. A high-performance Go Coordinator, a fine-tuned verifier model served via vLLM, and a Kubernetes operator are on the roadmap.

## License

Apache 2.0 © 2026 AgentRaft / ImaraCode LLC
