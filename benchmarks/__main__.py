"""CLI for the AgentRaft reliability benchmark.

    python -m benchmarks                      # full simulated benchmark
    python -m benchmarks --trials 1000        # more trials = tighter numbers
    python -m benchmarks --quick              # fast smoke run
    python -m benchmarks --live               # use the real configured verifier
    python -m benchmarks --live --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0

In ``--live`` mode the harness uses a real verifier and also prints the verifier
confusion table (recall per error class) — the artifact for the moat.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

# UTF-8 output for the table glyphs, even on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from .report import (  # noqa: E402
    headline,
    length_sweep,
    print_confusion,
    recall_sweep,
    write_results,
)


async def _main(args: argparse.Namespace) -> None:
    verifier = None
    conf = None

    if args.live:
        from agentraft.verifier import RulesVerifier, TieredVerifier
        from agentraft.verifier.llm import LLMVerifier
        from .confusion import measure_verifier

        llm = LLMVerifier(provider=args.provider, model=args.model)
        verifier = TieredVerifier(l1=RulesVerifier(), l2=llm)
        print(f"LIVE mode — verifier: {verifier.name} (L2={llm.name})")
        conf = await measure_verifier(llm, samples_per_class=args.confusion_samples)
        print_confusion(conf)
    else:
        print("SIMULATED mode — fault injection with a tunable verifier "
              "(use --live for a real model).")

    print(f"\nTrials per scenario: {args.trials}  ·  seed: 1234")

    comps = {
        "headline": await headline(args.trials, verifier),
        "length_sweep": await length_sweep(args.trials, verifier),
    }
    if not args.live:
        comps["recall_sweep"] = await recall_sweep(args.trials)

    path = write_results(comps, conf)
    print(f"\nResults written to {path}")
    print(
        "\nMethodology: controlled fault injection. Agents fail per-step at a set rate "
        "and emit taxonomy-typed bad outputs, so ground truth is known exactly. Runs go "
        "through the real AgentRaft Coordinator. 'silent corruption' = a wrong result "
        "shipped undetected — the metric AgentRaft is built to crush."
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="benchmarks", description="AgentRaft reliability benchmark")
    p.add_argument("--trials", type=int, default=400, help="trials per scenario (default 400)")
    p.add_argument("--quick", action="store_true", help="fast smoke run (50 trials)")
    p.add_argument("--live", action="store_true", help="use a real LLM/rules verifier")
    p.add_argument("--provider", default=None, help="bedrock | openai | anthropic | gemini")
    p.add_argument("--model", default=None, help="provider model id")
    p.add_argument("--confusion-samples", type=int, default=20, help="samples per class (live)")
    args = p.parse_args()
    if args.quick:
        args.trials = 50
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
