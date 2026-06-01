"""Core pipeline primitives: Task, Step, StepContext, Pipeline.

A pipeline is an ordered list of steps. Each step is an async (or sync) callable
that receives a :class:`StepContext` and returns an output. The Coordinator runs
each step, verifies its output, and only commits it to the checkpoint store once
the verifier passes it.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class Criticality(Enum):
    """How rigorously a step's output should be verified.

    The tiered verifier routes by this: LOW -> L1 rules only, MEDIUM -> small
    LLM verifier, HIGH -> large LLM verifier.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """A unit of work handed to a pipeline."""

    goal: str
    inputs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class StepContext:
    """Everything a step needs to do its work.

    Attributes:
        task:    the task being executed.
        outputs: outputs of all previously *verified* steps, keyed by step name.
        attempt: 0 on the first try, incremented on each retry.
        hint:    a typed correction hint from the verifier (set on retries only).
    """

    task: Task
    outputs: dict[str, Any]
    attempt: int = 0
    hint: Optional[str] = None

    @property
    def is_retry(self) -> bool:
        return self.attempt > 0


# A step function may be sync or async.
StepFn = Union[
    Callable[[StepContext], Awaitable[Any]],
    Callable[[StepContext], Any],
]


@dataclass
class Step:
    """A single stage in a pipeline."""

    name: str
    fn: StepFn
    goal: Optional[str] = None  # what this step should achieve, used by the verifier
    criticality: Criticality = Criticality.MEDIUM
    max_retries: Optional[int] = None  # overrides the global config when set

    async def execute(self, ctx: StepContext) -> Any:
        result = self.fn(ctx)
        if inspect.isawaitable(result):
            return await result
        return result


def step(
    name: Optional[str] = None,
    *,
    goal: Optional[str] = None,
    criticality: Criticality = Criticality.MEDIUM,
    max_retries: Optional[int] = None,
) -> Callable[[StepFn], Step]:
    """Decorator turning a function into a :class:`Step`.

    Example::

        @step(goal="Produce a 3-bullet summary of the sources", criticality=Criticality.HIGH)
        async def summarize(ctx):
            ...
    """

    def deco(fn: StepFn) -> Step:
        return Step(
            name=name or getattr(fn, "__name__", "step"),
            fn=fn,
            goal=goal,
            criticality=criticality,
            max_retries=max_retries,
        )

    return deco


@dataclass
class Pipeline:
    """An ordered collection of steps.

    Plain callables are accepted and wrapped into :class:`Step` automatically, so
    both of these work::

        Pipeline([research, analyse, draft])           # bare async functions
        Pipeline([Step("research", research, goal=...)])# explicit steps
    """

    steps: list[Step]
    name: str = "pipeline"

    def __post_init__(self) -> None:
        normalized: list[Step] = []
        for i, s in enumerate(self.steps):
            if isinstance(s, Step):
                normalized.append(s)
            elif callable(s):
                normalized.append(Step(name=getattr(s, "__name__", f"step_{i}"), fn=s))
            else:
                raise TypeError(
                    f"Pipeline steps must be Step instances or callables, got {type(s)!r}"
                )
        if not normalized:
            raise ValueError("Pipeline must contain at least one step")
        names = [s.name for s in normalized]
        if len(names) != len(set(names)):
            raise ValueError(f"Pipeline step names must be unique, got {names}")
        self.steps = normalized

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
