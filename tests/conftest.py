import pytest

from agentraft import Pipeline, Step, StepContext


@pytest.fixture
def simple_pipeline() -> Pipeline:
    async def a(ctx: StepContext) -> str:
        return "alpha"

    async def b(ctx: StepContext) -> str:
        return f"{ctx.outputs['a']}-beta"

    async def c(ctx: StepContext) -> str:
        return f"{ctx.outputs['b']}-gamma"

    return Pipeline([Step("a", a), Step("b", b), Step("c", c)])
