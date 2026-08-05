"""Engine contract: how ONE request sees the shared loop.

Pairs with test_batching.py (WHAT runs) and test_runners.py (HOW it runs). What
is graded here is the per-request lifecycle -- does it wait for the sentinel, does
multi-byte text survive streaming, does the slot come back when a client
disconnects -- none of which needs a model.

`drive()` stands in for run_loop with scripted tokens through the REAL Scheduler.
"""

import asyncio

import pytest

from batching import Scheduler
from engines import BatchingEngine, Completion

# Two token ids that only decode to a character together -- a stand-in for the
# CJK/emoji case where one character spans several BPE tokens.
PAIR_LEAD, PAIR_TAIL, PAIR_CHAR = 200, 201, "好"
EOS = 151645


class StubRunner:
    """A ModelRunner with no model. Execution is driven by the test, not by this."""

    eos_token_ids = {EOS}

    def encode(self, prompt):
        return [ord(c) for c in prompt] or [1]

    def decode(self, token_ids):
        out, i, ids = [], 0, list(token_ids)
        while i < len(ids):
            token = ids[i]
            if token == EOS:
                i += 1
            elif token == PAIR_LEAD:
                # An incomplete pair contributes NOTHING -- the whole point.
                if i + 1 < len(ids) and ids[i + 1] == PAIR_TAIL:
                    out.append(PAIR_CHAR)
                    i += 2
                else:
                    i += 1
            elif token == PAIR_TAIL:
                i += 1
            else:
                out.append(chr(ord("a") + token % 26))
                i += 1
        return "".join(out)

    def reset(self):
        pass

    def execute(self, batch):
        raise AssertionError("these tests drive the scheduler directly")


@pytest.fixture
def engine():
    return BatchingEngine(StubRunner(), Scheduler(max_batch_size=4, max_batch_tokens=1000))


async def drive(scheduler, script, ticks: int = 50):
    """Stand-in for run_loop: no runner, scripted tokens, the real Scheduler."""
    for tick in range(ticks):
        batch = scheduler.schedule()
        if batch:
            scheduler.postprocess(batch, [script(s, tick) for s in batch])
        await asyncio.sleep(0)
        if not scheduler.has_work():
            return


def letters(*ids):
    return StubRunner().decode(list(ids))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_generate_submits_to_the_scheduler(engine):
    task = asyncio.create_task(engine.generate("hi", 4, 0.0, None))
    await asyncio.sleep(0)

    assert engine.scheduler.has_work()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_generate_returns_the_accumulated_text(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    result = await engine.generate("hi", 3, 0.0, None)
    await loop

    assert isinstance(result, Completion)
    assert result.text == letters(3, 4, 5)


@pytest.mark.anyio
async def test_generate_reports_honest_counts_and_finish_reason(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    result = await engine.generate("hi", 3, 0.0, None)
    await loop

    assert result.completion_tokens == 3
    assert result.prompt_tokens == 2
    assert result.finish_reason == "length"


@pytest.mark.anyio
async def test_generate_stops_on_eos_and_says_stop(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: EOS if t else 5))

    result = await engine.generate("hi", 99, 0.0, None)
    await loop

    assert result.finish_reason == "stop"
    assert result.completion_tokens == 2, "EOS stays in output_ids so the count is honest"
    assert result.text == letters(5), "...but must not reach the JSON body"


@pytest.mark.anyio
async def test_generate_releases_the_slot_when_the_client_disconnects(engine):
    """FastAPI cancels the coroutine on disconnect. Without the try/finally this
    request keeps its batch slot and generates all 99 tokens for nobody."""
    task = asyncio.create_task(engine.generate("hi", 99, 0.0, None))
    await asyncio.sleep(0)
    engine.scheduler.schedule()
    assert engine.scheduler.num_running == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.scheduler.schedule() == []


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stream_yields_pieces_as_they_arrive(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert "".join(pieces) == letters(3, 4, 5)


@pytest.mark.anyio
async def test_stream_never_yields_an_empty_piece(engine):
    """An empty SSE chunk is pure noise on the wire."""
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert all(pieces)


@pytest.mark.anyio
async def test_stream_does_not_split_a_multi_token_character(engine):
    """One CJK char spans several BPE tokens; per-token decoding gives U+FFFD."""
    script = [PAIR_LEAD, PAIR_TAIL, 5]
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: script[t]))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert "".join(pieces) == PAIR_CHAR + letters(5)
    assert pieces.count(PAIR_CHAR) == 1


@pytest.mark.anyio
async def test_stream_matches_generate_under_the_same_script(engine):
    """Two code paths, one answer -- the Milestone 3 property, preserved."""
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))
    streamed = "".join([p async for p in engine.stream("hi", 3, 0.0, None)])
    await loop

    engine.scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))
    oneshot = await engine.generate("hi", 3, 0.0, None)
    await loop

    assert streamed == oneshot.text


@pytest.mark.anyio
async def test_stream_releases_the_slot_when_the_client_hangs_up(engine):
    """Streaming clients are the ones that disconnect early."""
    stream = engine.stream("hi", 99, 0.0, None)
    task = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    engine.scheduler.schedule()
    assert engine.scheduler.num_running == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()

    assert engine.scheduler.schedule() == []


# ---------------------------------------------------------------------------
# stats -- the metric that finally measures something
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_waiting_reflects_real_capacity_not_an_invented_ceiling(engine):
    """The point of the whole milestone.

    Three requests against max_batch_size=2: two run, one genuinely cannot start
    because the batch is full. That 1 is a measurement, not a semaphore's opinion.
    """
    engine.scheduler.max_batch_size = 2
    tasks = [asyncio.create_task(engine.generate("hi", 99, 0.0, None)) for _ in range(3)]
    await asyncio.sleep(0)
    engine.scheduler.schedule()

    assert engine.stats() == (2, 1)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
