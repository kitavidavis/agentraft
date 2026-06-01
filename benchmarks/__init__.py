"""AgentRaft reliability benchmark — measures the lift AgentRaft provides over an
unprotected pipeline, via controlled fault injection through the real Coordinator."""
from .faults import FaultConfig, GroundTruth, build_pipeline
from .harness import Aggregate, Comparison, Outcome, Scenario, evaluate, evaluate_sync
from .sim_verifier import SimulatedVerifier

__all__ = [
    "FaultConfig", "GroundTruth", "build_pipeline",
    "Scenario", "Comparison", "Aggregate", "Outcome", "evaluate", "evaluate_sync",
    "SimulatedVerifier",
]
