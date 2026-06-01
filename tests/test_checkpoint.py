import pytest

from agentraft import Checkpoint, InMemoryCheckpointStore


def _cp(task_id, idx, name, output):
    return Checkpoint(task_id=task_id, step_index=idx, step_name=name, output=output, attempt=0)


@pytest.mark.asyncio
async def test_save_and_latest():
    s = InMemoryCheckpointStore()
    await s.save(_cp("t1", 0, "a", "alpha"))
    await s.save(_cp("t1", 1, "b", "beta"))
    latest = await s.latest("t1")
    assert latest.step_name == "b"
    assert latest.output == "beta"


@pytest.mark.asyncio
async def test_history_and_outputs_map():
    s = InMemoryCheckpointStore()
    await s.save(_cp("t1", 0, "a", "alpha"))
    await s.save(_cp("t1", 1, "b", "beta"))
    assert len(await s.history("t1")) == 2
    assert await s.outputs("t1") == {"a": "alpha", "b": "beta"}


@pytest.mark.asyncio
async def test_rollback_discards_later_steps():
    s = InMemoryCheckpointStore()
    for i, (n, o) in enumerate([("a", "A"), ("b", "B"), ("c", "C")]):
        await s.save(_cp("t1", i, n, o))
    await s.rollback_to("t1", 0)  # keep only step 0
    hist = await s.history("t1")
    assert [c.step_name for c in hist] == ["a"]


@pytest.mark.asyncio
async def test_rollback_to_minus_one_clears():
    s = InMemoryCheckpointStore()
    await s.save(_cp("t1", 0, "a", "A"))
    await s.rollback_to("t1", -1)
    assert await s.history("t1") == []


@pytest.mark.asyncio
async def test_tasks_are_isolated():
    s = InMemoryCheckpointStore()
    await s.save(_cp("t1", 0, "a", "A"))
    await s.save(_cp("t2", 0, "a", "Z"))
    assert (await s.latest("t1")).output == "A"
    assert (await s.latest("t2")).output == "Z"
    await s.clear("t1")
    assert await s.latest("t1") is None
    assert (await s.latest("t2")).output == "Z"


def test_checkpoint_roundtrip_dict():
    cp = _cp("t1", 2, "draft", {"key": "value"})
    assert Checkpoint.from_dict(cp.to_dict()).output == {"key": "value"}
