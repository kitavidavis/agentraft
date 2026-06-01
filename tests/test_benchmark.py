"""Tests for the benchmark harness — deterministic, fully offline."""
import pytest

from benchmarks import Scenario, evaluate
from benchmarks.faults import FaultConfig


@pytest.mark.asyncio
async def test_agentraft_beats_baseline():
    """A decent verifier should lift success and cut silent corruption."""
    s = Scenario(n_steps=12, fault=FaultConfig(p_fail=0.12), recall=0.9, fpr=0.03)
    c = await evaluate(s, trials=120, seed=1)
    assert c.protected.success_rate > c.baseline.success_rate
    assert c.protected.silent_corruption_rate < c.baseline.silent_corruption_rate


@pytest.mark.asyncio
async def test_perfect_verifier_eliminates_silent_corruption():
    """recall=1.0, fpr=0.0 -> nothing bad should ever ship undetected."""
    s = Scenario(n_steps=15, fault=FaultConfig(p_fail=0.2), recall=1.0, fpr=0.0, max_retries=4)
    c = await evaluate(s, trials=120, seed=2)
    assert c.protected.silent_corruption_rate == 0.0


@pytest.mark.asyncio
async def test_baseline_reproduces_pn_decay():
    """Baseline success ≈ (1 - p_fail)^n, the reliability-compounding law."""
    n, p = 10, 0.1
    s = Scenario(n_steps=n, fault=FaultConfig(p_fail=p), recall=0.9)
    c = await evaluate(s, trials=600, seed=3)
    expected = (1 - p) ** n  # ~0.349
    assert abs(c.baseline.success_rate - expected) < 0.06


@pytest.mark.asyncio
async def test_higher_recall_reduces_corruption():
    """Monotonic-ish: better verifier recall -> less silent corruption."""
    base = FaultConfig(p_fail=0.15)
    low = await evaluate(Scenario(n_steps=15, fault=base, recall=0.5), trials=150, seed=4)
    high = await evaluate(Scenario(n_steps=15, fault=base, recall=0.99), trials=150, seed=4)
    assert high.protected.silent_corruption_rate <= low.protected.silent_corruption_rate


@pytest.mark.asyncio
async def test_outcomes_partition():
    """The three protected outcomes must sum to the trial count."""
    s = Scenario(n_steps=10, fault=FaultConfig(p_fail=0.15), recall=0.8)
    c = await evaluate(s, trials=100, seed=5)
    p = c.protected
    assert p.success + p.silent_corruption + p.caught_failure == p.trials


@pytest.mark.asyncio
async def test_agentraft_cheaper_than_full_rerun():
    s = Scenario(n_steps=20, fault=FaultConfig(p_fail=0.1), recall=0.9)
    c = await evaluate(s, trials=80, seed=6)
    assert c.cost_vs_full_rerun > 1.0  # AgentRaft is cheaper than naive full reruns
