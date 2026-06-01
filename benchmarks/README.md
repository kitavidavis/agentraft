# AgentRaft Benchmark

Measures the reliability lift AgentRaft provides over an unprotected agent pipeline —
and, crucially, shows how that lift depends on **verifier quality**. This is what turns
the "36% → 99%" headline from arithmetic into a measurement, and it's the eval you'd use
to justify investing in a better (fine-tuned) verifier.

```bash
python -m benchmarks                 # full simulated benchmark (400 trials/scenario)
python -m benchmarks --quick         # fast smoke run (50 trials)
python -m benchmarks --trials 1000   # tighter numbers
python -m benchmarks --live --provider bedrock \
    --model anthropic.claude-3-5-sonnet-20241022-v2:0   # measure a real verifier
```

Results are printed as tables and written to `benchmarks/results/results.json`.

---

## Methodology — controlled fault injection

This is a **simulation**, and we're explicit about that. We don't run live agents on
real tasks (that's expensive, non-deterministic, and hard to label). Instead:

1. Each step is an agent that **fails with a tunable probability** `p_fail`. On failure
   it emits an output corrupted in one of the five taxonomy classes (`GOAL_DRIFT`,
   `HALLUCINATION`, `INCOMPLETE`, `CONTRADICTION`, `SCOPE_CREEP`).
2. Because we *inject* the fault, we know the **ground-truth label of every output**.
   That's what lets us measure verifier recall and end-to-end silent-corruption rate
   exactly, instead of guessing.
3. Outputs are realistic *text*, not just a label — so the very same faulty pipeline can
   be judged by the ground-truth-aware simulated verifier **or** by a real LLM/rules
   verifier in `--live` mode.
4. Runs go through the **real AgentRaft `Coordinator`** — we're exercising the actual
   library, not a re-implementation.
5. Typed correction hints are modeled: a step's failure probability is multiplied by
   `retry_improvement` on each retry (hints make retries more likely to succeed).

Baseline (no AgentRaft) and protected (wrapped) trials are **paired on the same seed**,
so they see identical attempt-0 faults — a fair head-to-head.

### What we count

Each protected trial ends in one of three outcomes:

| Outcome | Meaning |
|---|---|
| **SUCCESS** | Completed; every committed output was correct. |
| **SILENT_CORRUPTION** | Completed, but a bad output slipped through (verifier miss). **The dangerous case** — a wrong result shipped with full confidence. |
| **CAUGHT_FAILURE** | Failed *safely* — a bad step was flagged but couldn't be recovered within the retry budget. Not a success, but nothing wrong shipped. |

The baseline can only land in SUCCESS or SILENT_CORRUPTION — with no verifier, every
corrupted run ships silently. **AgentRaft's core value is converting silent corruption
into either success or a safe, caught failure.**

---

## The three reports

### 1. Headline — baseline vs AgentRaft
Three representative tasks (short/reliable, medium, long/critical) compared head-to-head:
success rate, silent-corruption rate, the corruption-reduction factor, and cost vs a
naive "rerun the whole pipeline on any failure" strategy.

### 2. Pipeline-length sweep
The reliability-compounding law in action. Baseline success follows `(1 − p_fail)ⁿ`;
AgentRaft stays roughly flat as the pipeline grows. At per-step reliability 0.90 the
baseline column is exactly:

| steps | baseline success = 0.9ⁿ |
|------:|------------------------:|
| 5 | 59.0% |
| 10 | 34.9% |
| 15 | 20.6% |
| 20 | 12.2% |

(These are analytic — the benchmark reproduces them empirically, then shows AgentRaft
holding the line above them. Run it to fill in the AgentRaft column for your machine.)

### 3. Verifier-quality sweep — *the moat*
Holds the pipeline fixed (15 steps) and sweeps simulated verifier recall
(0.50 → 0.99). End-to-end reliability tracks verifier recall almost directly — which is
the quantitative argument for a better verifier model. **This is why the fine-tuned
3–7B verifier is the defensible asset, not the orchestration loop.**

---

## Live mode — measuring a *real* verifier

```bash
python -m benchmarks --live --provider bedrock --model meta.llama3-1-70b-instruct-v1:0
```

In `--live` mode the harness swaps the simulated verifier for a real one and prints a
**confusion table**: for each injected error class, what fraction did the verifier catch,
and did it assign the right class?

This is the bridge to training a verifier. Expect the **rules-only** L1 verifier to catch
`INCOMPLETE` (it detects empty/truncated/error outputs) but **miss the semantic classes**
(`GOAL_DRIFT`, `HALLUCINATION`, `CONTRADICTION`, `SCOPE_CREEP`) — which is precisely the
empirical case for a semantic LLM/fine-tuned verifier. The Bedrock/OpenAI/Anthropic/Gemini
verifiers should lift recall on those classes substantially.

The confusion result (per-class recall, class accuracy, false-positive rate) is the exact
label-quality signal you'd track while building a custom verifier model.

---

## Knobs

All configurable via `Scenario` (`benchmarks/harness.py`) and `FaultConfig`
(`benchmarks/faults.py`):

| Knob | Default | Meaning |
|---|---|---|
| `n_steps` | 10 | pipeline length |
| `p_fail` | 0.10 | per-step failure probability (1 − per-step reliability) |
| `error_dist` | weighted mix | which error classes faults draw from |
| `retry_improvement` | 0.45 | failure-prob multiplier per retry (hint effectiveness) |
| `max_retries` | 3 | per-step retry budget |
| `recall` | 0.90 | simulated verifier true-positive rate |
| `fpr` | 0.03 | simulated verifier false-positive rate |
| `verify_cost_ratio` | 0.08 | cost of one verification ÷ one generation |

---

## Honest limitations

- It's a **simulation**. The headline percentages depend on the injected `p_fail`,
  `recall`, and `retry_improvement` — they demonstrate the *mechanism* and its dependence
  on verifier quality, not a claim about any specific real workload.
- The `retry_improvement` assumption (that typed hints help) is a model, not yet a
  measured property of real agents. Validating it on live tasks is future work.
- "Strict propagation" is assumed: any corrupted committed step makes the final result
  wrong. Real pipelines are sometimes more forgiving; this is the conservative view.

The path to a non-simulated benchmark: run real agent pipelines on a labeled multi-step
task suite (e.g. GAIA-style tasks) and measure the same three outcomes. The harness is
structured so that swapping the faulty agents for real ones — and `--live` for the
verifier — is the only change required.
