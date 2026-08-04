"""Grader for the Milestone 4 Day 12 wiring (docs/continuous-batching-tutorial.md).

Day 11 built a scheduler that nothing calls. This grades the part that connects
it: BatchingEngine as a per-request view onto the shared loop, cancellation when
a client disconnects, teardown when a tick dies, and the metrics switchover that
turns vllm:num_requests_waiting from a simulation into a measurement.

    TODO 7   Scheduler.abort        (batching.py)
    TODO 8   Scheduler.fail_all     (batching.py)
    TODO 9   BatchingEngine.generate
    TODO 10  BatchingEngine.stream
    TODO 11  engine_stats

Run one at a time with `pytest test_engines_batching.py -m todo9 -v`.

No GPU, no weights, no transformers, and deliberately no real run_loop: `drive()`
below stands in for it by emitting scripted tokens through the REAL Scheduler.
What is under test is the engine's contract with the loop -- does it wait for the
sentinel, does it decode multi-byte characters correctly, does it release the slot
when the client vanishes -- none of which needs a forward pass to grade.
"""

import asyncio

import pytest

pytest.importorskip("torch", reason="batching.py imports torch")

from batching import Scheduler, Sequence  # noqa: E402
from engines import BatchingEngine, Completion, MockEngine, engine_stats  # noqa: E402

# Two token ids that only decode to a character when BOTH are present -- a stand-in
# for the CJK/emoji case where one character spans several BPE tokens.
PAIR_LEAD, PAIR_TAIL, PAIR_CHAR = 200, 201, "好"
EOS = 151645
SPECIAL = {EOS, 151643}


class StubTokenizer:
    """Enough tokenizer surface for build_prompt + prompt_to_sequence + decode."""

    eos_token_id = EOS
    pad_token_id = 151643

    def apply_chat_template(self, messages, **kwargs):
        return f"<im>{messages[0]['content']}</im>"

    def __call__(self, text, **kwargs):
        # One token per character, offset into a boring range.
        return type("Enc", (), {"input_ids": [ord(c) % 50 + 1 for c in text]})()

    def decode(self, ids, skip_special_tokens=False):
        out, i, ids = [], 0, list(ids)
        while i < len(ids):
            token = ids[i]
            if skip_special_tokens and token in SPECIAL:
                i += 1
            elif token == PAIR_LEAD:
                # Incomplete pair contributes NOTHING -- this is the whole point.
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


class StubModel:
    """Only generation_config matters -- resolve_eos_ids is all the engine reads."""

    generation_config = type("GC", (), {"eos_token_id": [EOS, 151643]})()


@pytest.fixture
def engine():
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    return BatchingEngine(StubModel(), StubTokenizer(), scheduler)


async def drive(scheduler: Scheduler, script, ticks: int = 50, delay: float = 0):
    """Stand-in for run_loop: no model, scripted tokens, the REAL Scheduler.

    `script(seq, tick)` returns the token id that sequence gets this tick.
    """
    for tick in range(ticks):
        batch = scheduler.schedule()
        if batch:
            scheduler.postprocess(batch, [script(s, tick) for s in batch])
        await asyncio.sleep(delay)
        if not scheduler.has_work():
            return


def letters(*ids):
    """The text StubTokenizer.decode produces for these output ids."""
    return StubTokenizer().decode(list(ids), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# TODO 7 -- Scheduler.abort
# ---------------------------------------------------------------------------


@pytest.mark.todo7
def test_abort_marks_a_running_sequence_finished():
    scheduler = Scheduler()
    seq = Sequence(prompt_ids=[1, 2], max_new_tokens=99)
    scheduler.add(seq)
    scheduler.schedule()

    scheduler.abort(seq)

    assert seq.finish_reason == "cancelled"


@pytest.mark.todo7
def test_abort_removes_a_still_waiting_sequence_from_the_queue():
    """schedule() only filters `running` -- a cancelled request left in `waiting`
    would still be admitted later and generate for a client that is long gone."""
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    running = Sequence(prompt_ids=[1], max_new_tokens=99)
    queued = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(running)
    scheduler.add(queued)
    scheduler.schedule()          # `running` is admitted, `queued` waits

    scheduler.abort(queued)

    assert scheduler.num_waiting == 0
    assert queued.id not in {s.id for s in scheduler.schedule()}


@pytest.mark.todo7
def test_abort_wakes_a_blocked_consumer():
    scheduler = Scheduler()
    seq = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(seq)

    scheduler.abort(seq)

    assert seq.queue.get_nowait() is None


@pytest.mark.todo7
def test_abort_accepts_a_custom_reason():
    scheduler = Scheduler()
    seq = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(seq)

    scheduler.abort(seq, reason="error")

    assert seq.finish_reason == "error"


@pytest.mark.todo7
def test_abort_is_safe_on_an_already_finished_sequence():
    """The engine's `finally` calls this unconditionally on the happy path."""
    scheduler = Scheduler()
    seq = Sequence(prompt_ids=[1], max_new_tokens=1)
    scheduler.add(seq)
    scheduler.schedule()
    scheduler.postprocess([seq], [7])
    assert seq.finish_reason == "length"

    scheduler.abort(seq)

    assert seq.finish_reason == "length", "must not overwrite a real finish reason"


@pytest.mark.todo7
def test_abort_frees_the_slot_for_the_next_request():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    first = Sequence(prompt_ids=[1], max_new_tokens=99)
    second = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(first)
    scheduler.add(second)
    scheduler.schedule()

    scheduler.abort(first)

    assert [s.id for s in scheduler.schedule()] == [second.id]


# ---------------------------------------------------------------------------
# TODO 8 -- Scheduler.fail_all
# ---------------------------------------------------------------------------


@pytest.mark.todo8
def test_fail_all_marks_running_and_waiting_alike():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    running = Sequence(prompt_ids=[1], max_new_tokens=99)
    queued = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(running)
    scheduler.add(queued)
    scheduler.schedule()

    scheduler.fail_all(RuntimeError("CUDA out of memory"))

    assert running.finish_reason == "error"
    assert queued.finish_reason == "error"


@pytest.mark.todo8
def test_fail_all_wakes_every_waiter():
    """Without this, a dead loop leaves every request hung on queue.get() forever."""
    scheduler = Scheduler()
    seqs = [Sequence(prompt_ids=[1], max_new_tokens=99) for _ in range(3)]
    for s in seqs:
        scheduler.add(s)
    scheduler.schedule()

    scheduler.fail_all(RuntimeError("boom"))

    for s in seqs:
        assert s.queue.get_nowait() is None


@pytest.mark.todo8
def test_fail_all_empties_the_scheduler():
    scheduler = Scheduler()
    scheduler.add(Sequence(prompt_ids=[1], max_new_tokens=99))
    scheduler.schedule()
    scheduler.add(Sequence(prompt_ids=[1], max_new_tokens=99))

    scheduler.fail_all(RuntimeError("boom"))

    assert scheduler.num_running == 0
    assert scheduler.num_waiting == 0
    assert not scheduler.has_work()


@pytest.mark.todo8
def test_fail_all_returns_what_it_failed():
    scheduler = Scheduler()
    seq = Sequence(prompt_ids=[1], max_new_tokens=99)
    scheduler.add(seq)

    assert [s.id for s in scheduler.fail_all(RuntimeError("boom"))] == [seq.id]


@pytest.mark.todo8
def test_fail_all_does_not_raise():
    """run_loop decides what to do with the exception; this only tears down."""
    scheduler = Scheduler()
    scheduler.add(Sequence(prompt_ids=[1], max_new_tokens=99))

    scheduler.fail_all(RuntimeError("boom"))  # must not propagate


# ---------------------------------------------------------------------------
# TODO 9 -- BatchingEngine.generate
# ---------------------------------------------------------------------------


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_submits_to_the_scheduler(engine):
    task = asyncio.create_task(engine.generate("hi", 4, 0.0, None))
    await asyncio.sleep(0)

    assert engine.scheduler.has_work(), "the request must reach the scheduler"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_returns_the_accumulated_text(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    result = await engine.generate("hi", 3, 0.0, None)
    await loop

    assert isinstance(result, Completion)
    assert result.text == letters(3, 4, 5)


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_reports_honest_counts_and_finish_reason(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    result = await engine.generate("hi", 3, 0.0, None)
    await loop

    assert result.completion_tokens == 3
    assert result.prompt_tokens > 0
    assert result.finish_reason == "length"


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_stops_on_eos_and_says_stop(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: EOS if t else 5))

    result = await engine.generate("hi", 99, 0.0, None)
    await loop

    assert result.finish_reason == "stop"
    assert result.completion_tokens == 2


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_strips_special_tokens_from_the_text(engine):
    """EOS stays in output_ids so the count is honest, but must not reach the JSON."""
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: EOS if t else 5))

    result = await engine.generate("hi", 99, 0.0, None)
    await loop

    assert result.text == letters(5)


@pytest.mark.todo9
@pytest.mark.anyio
async def test_generate_releases_the_slot_when_the_client_disconnects(engine):
    """FastAPI cancels the coroutine on disconnect; the sequence must not survive it.

    Without the try/finally, this request keeps its batch slot and generates all
    99 tokens for a client that is already gone.
    """
    task = asyncio.create_task(engine.generate("hi", 99, 0.0, None))
    await asyncio.sleep(0)
    engine.scheduler.schedule()
    assert engine.scheduler.num_running == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.scheduler.schedule() == [], "the abandoned sequence must be dropped"


# ---------------------------------------------------------------------------
# TODO 10 -- BatchingEngine.stream
# ---------------------------------------------------------------------------


@pytest.mark.todo10
@pytest.mark.anyio
async def test_stream_yields_pieces_as_they_arrive(engine):
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert "".join(pieces) == letters(3, 4, 5)


@pytest.mark.todo10
@pytest.mark.anyio
async def test_stream_never_yields_an_empty_piece(engine):
    """An empty SSE chunk is pure noise on the wire."""
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: 3 + t))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert all(pieces), f"empty chunk in {pieces!r}"


@pytest.mark.todo10
@pytest.mark.anyio
async def test_stream_does_not_split_a_multi_token_character(engine):
    """One CJK char = several BPE tokens. Decoding per token gives U+FFFD.

    The lead token alone decodes to nothing; the character must appear exactly
    once, whole, when its tail token arrives.
    """
    script = [PAIR_LEAD, PAIR_TAIL, 5]
    loop = asyncio.create_task(drive(engine.scheduler, lambda s, t: script[t]))

    pieces = [p async for p in engine.stream("hi", 3, 0.0, None)]
    await loop

    assert "".join(pieces) == PAIR_CHAR + letters(5)
    assert pieces.count(PAIR_CHAR) == 1


@pytest.mark.todo10
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


@pytest.mark.todo10
@pytest.mark.anyio
async def test_stream_releases_the_slot_when_the_client_hangs_up(engine):
    """Streaming clients are the ones that disconnect early. Closing the async
    generator must abort the sequence, not leak it."""
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
# TODO 11 -- engine_stats
# ---------------------------------------------------------------------------


@pytest.mark.todo11
def test_stats_come_from_the_engine_when_it_owns_concurrency(engine):
    engine.scheduler.add(Sequence(prompt_ids=[1], max_new_tokens=9))
    engine.scheduler.add(Sequence(prompt_ids=[1], max_new_tokens=9))
    engine.scheduler.max_batch_size = 1
    engine.scheduler.schedule()

    assert engine_stats(engine, {"running": 99, "waiting": 99}) == (1, 1)


@pytest.mark.todo11
def test_stats_come_from_the_runtime_dict_for_the_other_engines():
    """Mock/NanoGPT/Qwen still let main.py's semaphore do the counting."""
    assert engine_stats(MockEngine("m"), {"running": 3, "waiting": 2}) == (3, 2)


@pytest.mark.todo11
def test_the_batching_engine_declares_that_it_owns_concurrency(engine):
    assert engine.owns_concurrency is True
    assert MockEngine("m").owns_concurrency is False


@pytest.mark.todo11
@pytest.mark.anyio
async def test_waiting_reflects_real_capacity_not_an_invented_ceiling(engine):
    """The point of the whole milestone.

    Three requests against max_batch_size=2: two run, one genuinely cannot start
    because the KV cache has no room for it. That 1 is a measurement.
    """
    engine.scheduler.max_batch_size = 2
    tasks = [
        asyncio.create_task(engine.generate("hi", 99, 0.0, None)) for _ in range(3)
    ]
    await asyncio.sleep(0)
    engine.scheduler.schedule()

    assert engine_stats(engine, {}) == (2, 1)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
