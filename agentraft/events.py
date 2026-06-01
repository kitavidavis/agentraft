"""Protocol events emitted by the Coordinator.

Subscribe via ``AgentRaftConfig.on_event`` to drive live monitors, dashboards, or
structured logs. Events mirror the lifecycle a step goes through:
running -> verifying -> committed | (failed -> rollback -> retry).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .errors import VerificationResult


class EventType(str, Enum):
    RUN_START = "run_start"
    STEP_START = "step_start"
    STEP_VERIFYING = "step_verifying"
    STEP_COMMITTED = "step_committed"
    STEP_FAILED = "step_failed"
    STEP_ROLLBACK = "step_rollback"
    STEP_RETRY = "step_retry"
    RUN_SUCCESS = "run_success"
    RUN_FAILED = "run_failed"
    BREAKER_OPEN = "breaker_open"


@dataclass
class Event:
    type: EventType
    task_id: str
    step_index: Optional[int] = None
    step_name: Optional[str] = None
    attempt: int = 0
    verification: Optional[VerificationResult] = None
    detail: Any = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "attempt": self.attempt,
            "verification": self.verification.to_dict() if self.verification else None,
            "detail": self.detail,
        }
