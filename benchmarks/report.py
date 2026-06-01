"""Run the standard benchmark and render reports (stdout tables + JSON/CSV)."""
from __future__ import annotations

import json
import os
from typing import Optional

from agentraft.verifier.base import Verifier

from .confusion import ConfusionResult
from .faults import FaultConfig
from .harness import Comparison, Scenario, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _x(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:,.1f}×"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


# ── Reports ───────────────────────────────────────────────────────────────────
async def headline(trials: int, verifier: Optional[Verifier]) -> list[Comparison]:
    scenarios = [
        Scenario(n_steps=5,  fault=FaultConfig(p_fail=0.05), recall=0.9, label="short / reliable"),
        Scenario(n_steps=10, fault=FaultConfig(p_fail=0.10), recall=0.9, label="medium"),
        Scenario(n_steps=20, fault=FaultConfig(p_fail=0.10), recall=0.9, label="long / critical"),
    ]
    comps = [await evaluate(s, trials=trials, verifier=verifier) for s in scenarios]
    rows = []
    for c in comps:
        per_step = 1 - c.scenario.fault.p_fail
        rows.append([
            c.scenario.label,
            str(c.scenario.n_steps),
            _pct(per_step),
            _pct(c.baseline.success_rate),
            _pct(c.protected.success_rate),
            _pct(c.baseline.silent_corruption_rate),
            _pct(c.protected.silent_corruption_rate),
            _x(c.corruption_reduction),
            _x(c.cost_vs_full_rerun),
        ])
    print("\n## Headline — baseline vs AgentRaft  (verifier recall=0.90, fpr=0.03)\n")
    print(_table(
        ["task", "steps", "per-step", "base ✓", "AR ✓",
         "base corrupt", "AR corrupt", "corrupt ↓", "vs full-rerun cost"],
        rows,
    ))
    return comps


async def length_sweep(trials: int, verifier: Optional[Verifier]) -> list[Comparison]:
    comps = []
    for n in (5, 10, 15, 20):
        s = Scenario(n_steps=n, fault=FaultConfig(p_fail=0.10), recall=0.9)
        comps.append(await evaluate(s, trials=trials, verifier=verifier))
    rows = [[
        str(c.scenario.n_steps),
        _pct(c.baseline.success_rate),
        _pct(c.protected.success_rate),
        _pct(c.protected.silent_corruption_rate),
        _pct(c.protected.caught_failure_rate),
    ] for c in comps]
    print("\n## Pipeline-length sweep  (per-step reliability 0.90, recall=0.90)\n")
    print("Baseline follows p^n decay; AgentRaft stays flat.\n")
    print(_table(["steps", "baseline ✓", "AgentRaft ✓", "AR silent corrupt", "AR caught-fail"], rows))
    return comps


async def recall_sweep(trials: int) -> list[Comparison]:
    comps = []
    for r in (0.5, 0.7, 0.9, 0.99):
        s = Scenario(n_steps=15, fault=FaultConfig(p_fail=0.10), recall=r, fpr=0.03)
        comps.append(await evaluate(s, trials=trials))
    rows = [[
        f"{c.scenario.recall:.2f}",
        _pct(c.protected.success_rate),
        _pct(c.protected.silent_corruption_rate),
        _pct(c.protected.caught_failure_rate),
    ] for c in comps]
    print("\n## Verifier-quality sweep  (15 steps, per-step reliability 0.90)\n")
    print("This is the moat: end-to-end reliability tracks verifier recall.\n")
    print(_table(["verifier recall", "AgentRaft ✓", "silent corrupt", "caught-fail"], rows))
    return comps


def print_confusion(conf: ConfusionResult) -> None:
    print(f"\n## Verifier confusion — {conf.verifier_name}\n")
    rows = [[
        cls.value,
        _pct(stat.recall),
        _pct(stat.class_accuracy),
        f"{stat.caught}/{stat.total}",
    ] for cls, stat in conf.per_class.items()]
    print(_table(["injected error", "recall", "class accuracy", "caught"], rows))
    print(f"\noverall recall: {_pct(conf.overall_recall)}   "
          f"false-positive rate (on good outputs): {_pct(conf.false_positive_rate)}")


def write_results(comps: dict[str, list[Comparison]], conf: Optional[ConfusionResult] = None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload: dict = {}
    for section, items in comps.items():
        payload[section] = [{
            "label": c.scenario.label,
            "n_steps": c.scenario.n_steps,
            "p_fail": c.scenario.fault.p_fail,
            "recall": c.scenario.recall,
            "baseline_success": c.baseline.success_rate,
            "protected_success": c.protected.success_rate,
            "baseline_silent_corruption": c.baseline.silent_corruption_rate,
            "protected_silent_corruption": c.protected.silent_corruption_rate,
            "protected_caught_failure": c.protected.caught_failure_rate,
            "corruption_reduction": c.corruption_reduction,
            "cost_vs_full_rerun": c.cost_vs_full_rerun,
        } for c in items]
    if conf:
        payload["verifier_confusion"] = {
            "verifier": conf.verifier_name,
            "overall_recall": conf.overall_recall,
            "false_positive_rate": conf.false_positive_rate,
            "per_class": {cls.value: {"recall": st.recall, "class_accuracy": st.class_accuracy}
                          for cls, st in conf.per_class.items()},
        }
    path = os.path.join(RESULTS_DIR, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
