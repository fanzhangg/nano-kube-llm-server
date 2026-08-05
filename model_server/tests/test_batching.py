"""Scheduler contract: WHAT runs, and when.

Pairs with test_runners.py (HOW a batch becomes tokens) and test_engines.py (how
one request sees the shared loop).

Deliberately torch-free, like the module it grades. If this file ever needs
`import torch`, something model-specific has leaked into the scheduler -- and the
150MB mock image would stop being buildable.
"""

import asyncio

import pytest

from batching import Scheduler, Sequence, prompt_to_sequence, run_loop


def seq(prompt_len: int = 4, max_new_tokens: int = 4, **kw) -> Sequence:
    return Sequence(
        prompt_ids=list(range(1, prompt_len + 1)),
        max_new_tokens=max_new_tokens,
        temperature=kw.pop("temperature", 0.0),
        top_k=kw.pop("top_k", None),
        **kw,
    )


class FakeRunner:
    """Records the ForwardBatch it was handed. No model, no torch."""

    eos_token_ids: set[int] = set()

    def __init__(self, script=None):
        self.script = script or (lambda call, row: 7)
        self.calls = []
        self.resets = 0

    def encode(self, prompt):
        return [ord(c) for c in prompt] or [1]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def reset(self):
        self.resets += 1

    def execute(self, batch):
        self.calls.append(batch)
        index = len(self.calls) - 1
        return [self.script(index, row) for row in range(len(batch.seqs))]


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------


def test_admits_up_to_the_batch_size():
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=1000)
    for _ in range(3):
        scheduler.add(seq())

    assert len(scheduler.schedule()) == 2
    assert scheduler.num_waiting == 1


def test_admits_in_fifo_order():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    first, second = seq(), seq()
    scheduler.add(first)
    scheduler.add(second)

    assert [s.id for s in scheduler.schedule()] == [first.id]


def test_respects_the_token_budget():
    scheduler = Scheduler(max_batch_size=10, max_batch_tokens=10)
    scheduler.add(seq(6))
    scheduler.add(seq(6))

    assert len(scheduler.schedule()) == 1, "6 fits, 6+6=12 does not"


def test_stops_at_the_first_request_that_does_not_fit():
    """FIFO, not best-fit. Packing the small one in first starves the big one."""
    scheduler = Scheduler(max_batch_size=10, max_batch_tokens=5)
    scheduler.add(seq(4))
    scheduler.schedule()
    big, small = seq(9), seq(1)
    scheduler.add(big)
    scheduler.add(small)

    assert len(scheduler.schedule()) == 1
    assert [s.id for s in scheduler.waiting] == [big.id, small.id]


def test_admits_an_oversized_prompt_into_an_empty_batch():
    """Refusing it deadlocks: it can never reach the front of an emptier queue."""
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=5)
    scheduler.add(seq(50))

    assert len(scheduler.schedule()) == 1


def test_drops_finished_sequences_and_backfills_the_same_tick():
    """The whole point of *continuous* batching."""
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=1000)
    a, b, c = seq(), seq(), seq()
    for s in (a, b, c):
        scheduler.add(s)
    scheduler.schedule()
    a.finish_reason = "stop"

    ids = {s.id for s in scheduler.schedule()}

    assert a.id not in ids
    assert c.id in ids


def test_idle_scheduler_schedules_nothing():
    scheduler = Scheduler()

    assert scheduler.schedule() == []
    assert not scheduler.has_work()


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def test_each_token_lands_on_its_own_sequence_and_queue():
    scheduler = Scheduler()
    a, b = seq(), seq()

    scheduler.postprocess([a, b], [5, 6])

    assert (a.output_ids, b.output_ids) == ([5], [6])
    assert (a.queue.get_nowait(), b.queue.get_nowait()) == (5, 6)


def test_eos_finishes_with_stop():
    scheduler = Scheduler()
    s = seq(max_new_tokens=10, eos_token_ids={2})

    assert [f.id for f in scheduler.postprocess([s], [2])] == [s.id]
    assert s.finish_reason == "stop"


def test_hitting_max_new_tokens_finishes_with_length():
    scheduler = Scheduler()
    s = seq(max_new_tokens=2, eos_token_ids={99})

    assert scheduler.postprocess([s], [5]) == []
    scheduler.postprocess([s], [6])

    assert s.finish_reason == "length"


def test_eos_on_the_final_allowed_token_reports_stop():
    """Check EOS before the length limit, or a clean stop reads as truncation."""
    scheduler = Scheduler()
    s = seq(max_new_tokens=1, eos_token_ids={2})

    scheduler.postprocess([s], [2])

    assert s.finish_reason == "stop"


def test_finished_sequence_gets_an_end_of_stream_sentinel():
    """An empty queue means "not yet"; None means "never again"."""
    scheduler = Scheduler()
    s = seq(max_new_tokens=1, eos_token_ids={99})

    scheduler.postprocess([s], [5])

    assert s.queue.get_nowait() == 5
    assert s.queue.get_nowait() is None


def test_unfinished_sequence_gets_no_sentinel():
    scheduler = Scheduler()
    s = seq(max_new_tokens=5, eos_token_ids={99})

    scheduler.postprocess([s], [5])
    s.queue.get_nowait()

    assert s.queue.empty()


def test_one_sequence_finishing_does_not_disturb_the_others():
    scheduler = Scheduler()
    done = seq(max_new_tokens=1, eos_token_ids={99})
    alive = seq(max_new_tokens=9, eos_token_ids={99})

    assert [f.id for f in scheduler.postprocess([done, alive], [1, 2])] == [done.id]
    assert alive.finish_reason == ""


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


def test_abort_marks_a_running_sequence():
    scheduler = Scheduler()
    s = seq(max_new_tokens=99)
    scheduler.add(s)
    scheduler.schedule()

    scheduler.abort(s)

    assert s.finish_reason == "cancelled"


def test_abort_removes_a_still_waiting_sequence():
    """schedule() only filters `running` -- a cancelled request left in `waiting`
    would still be admitted later and generate for a client long gone."""
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    running, queued = seq(max_new_tokens=99), seq(max_new_tokens=99)
    scheduler.add(running)
    scheduler.add(queued)
    scheduler.schedule()

    scheduler.abort(queued)

    assert scheduler.num_waiting == 0
    assert queued.id not in {s.id for s in scheduler.schedule()}


def test_abort_wakes_a_blocked_consumer():
    scheduler = Scheduler()
    s = seq(max_new_tokens=99)
    scheduler.add(s)

    scheduler.abort(s)

    assert s.queue.get_nowait() is None


def test_abort_is_safe_on_an_already_finished_sequence():
    """The engine's `finally` calls this unconditionally on the happy path."""
    scheduler = Scheduler()
    s = seq(max_new_tokens=1)
    scheduler.add(s)
    scheduler.schedule()
    scheduler.postprocess([s], [7])

    scheduler.abort(s)

    assert s.finish_reason == "length", "must not overwrite a real finish reason"


def test_abort_frees_the_slot():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    first, second = seq(max_new_tokens=99), seq(max_new_tokens=99)
    scheduler.add(first)
    scheduler.add(second)
    scheduler.schedule()

    scheduler.abort(first)

    assert [s.id for s in scheduler.schedule()] == [second.id]


# ---------------------------------------------------------------------------
# fail_all
# ---------------------------------------------------------------------------


def test_fail_all_marks_running_and_waiting_alike():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    running, queued = seq(max_new_tokens=99), seq(max_new_tokens=99)
    scheduler.add(running)
    scheduler.add(queued)
    scheduler.schedule()

    scheduler.fail_all(RuntimeError("CUDA out of memory"))

    assert running.finish_reason == "error"
    assert queued.finish_reason == "error"


def test_fail_all_wakes_every_waiter():
    """Without this a dead loop leaves every request hung on get() forever."""
    scheduler = Scheduler()
    seqs = [seq(max_new_tokens=99) for _ in range(3)]
    for s in seqs:
        scheduler.add(s)
    scheduler.schedule()

    scheduler.fail_all(RuntimeError("boom"))

    assert all(s.queue.get_nowait() is None for s in seqs)


def test_fail_all_empties_the_scheduler_and_returns_the_casualties():
    scheduler = Scheduler()
    s = seq(max_new_tokens=99)
    scheduler.add(s)
    scheduler.schedule()

    assert [f.id for f in scheduler.fail_all(RuntimeError("boom"))] == [s.id]
    assert not scheduler.has_work()


# ---------------------------------------------------------------------------
# run_loop -- the WHAT/HOW boundary
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_loop_marks_the_first_tick_as_prefill():
    runner, scheduler = FakeRunner(), Scheduler()
    scheduler.add(seq(3, max_new_tokens=1, eos_token_ids={99}))

    task = asyncio.create_task(run_loop(runner, scheduler))
    await asyncio.sleep(0.05)
    task.cancel()

    assert runner.calls[0].is_prefill is True


@pytest.mark.anyio
async def test_loop_marks_a_stable_batch_as_decode():
    runner, scheduler = FakeRunner(), Scheduler()
    scheduler.add(seq(3, max_new_tokens=3, eos_token_ids={99}))

    task = asyncio.create_task(run_loop(runner, scheduler))
    await asyncio.sleep(0.05)
    task.cancel()

    assert [c.is_prefill for c in runner.calls[:3]] == [True, False, False]


@pytest.mark.anyio
async def test_membership_change_forces_prefill_again():
    """The runner's cache is indexed by row position; change the rows, drop it."""
    runner, scheduler = FakeRunner(), Scheduler()
    scheduler.add(seq(3, max_new_tokens=9, eos_token_ids={99}))

    task = asyncio.create_task(run_loop(runner, scheduler))
    await asyncio.sleep(0.03)
    before = len(runner.calls)
    scheduler.add(seq(4, max_new_tokens=9, eos_token_ids={99}))
    await asyncio.sleep(0.03)
    task.cancel()

    assert runner.calls[before].is_prefill is True


@pytest.mark.anyio
async def test_a_dying_tick_fails_every_request_instead_of_hanging():
    """The failure mode this prevents is a server that answers nothing, forever."""

    class Exploding(FakeRunner):
        def execute(self, batch):
            raise RuntimeError("CUDA out of memory")

    scheduler = Scheduler()
    s = seq(max_new_tokens=99)
    scheduler.add(s)

    task = asyncio.create_task(run_loop(Exploding(), scheduler))
    await asyncio.sleep(0.05)

    assert s.finish_reason == "error"
    assert await asyncio.wait_for(s.queue.get(), timeout=1) is None
    with pytest.raises(RuntimeError):
        await task


@pytest.mark.anyio
async def test_a_row_that_fails_alone_does_not_take_the_batch_with_it():
    """One request's fault costs one request. The blast radius IS the contract.

    A None token id means the runner could attribute the failure to a single row
    -- a bad sampling param, NaN logits in one row. Failing the batch there
    would let any one client take out the seven strangers sharing its forward
    pass, which is how {"top_k": "5"} used to kill the whole server. Contrast
    test_a_dying_tick_fails_every_request_instead_of_hanging: a raise means the
    runner could NOT attribute it, and then everybody does go down.
    """
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    doomed, bystander = seq(max_new_tokens=2), seq(max_new_tokens=2)

    class OneBadRow(FakeRunner):
        # Keyed on the SEQUENCE, not the row index: once the casualty is
        # dropped the bystander inherits row 0, and an index-keyed fake would
        # then kill it too -- and call that a failure of the code under test.
        def execute(self, batch):
            self.calls.append(batch)
            return [None if s.id == doomed.id else 7 for s in batch.seqs]

    scheduler.add(doomed)
    scheduler.add(bystander)

    task = asyncio.create_task(run_loop(OneBadRow(), scheduler))
    await asyncio.sleep(0.05)

    assert doomed.finish_reason == "error"
    assert await asyncio.wait_for(doomed.queue.get(), timeout=1) is None
    assert bystander.finish_reason == "length", "a bystander lost its request"
    assert bystander.output_ids == [7, 7]
    assert not task.done(), "the loop must survive a single row's failure"

    task.cancel()


@pytest.mark.anyio
async def test_a_failed_row_forces_a_prefill_on_the_next_tick():
    """Dropping a row invalidates the cache: it is indexed by row POSITION.

    Keep decoding after a casualty and every surviving row reads another
    sequence's KV entries -- fluent output, wrong context, no error anywhere.
    """
    class FirstTickHasACasualty(FakeRunner):
        def execute(self, batch):
            self.calls.append(batch)
            first_tick = len(self.calls) == 1
            return [None if (first_tick and i == 0) else 7
                    for i in range(len(batch.seqs))]

    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    scheduler.add(seq(max_new_tokens=1))
    scheduler.add(seq(max_new_tokens=3))

    runner = FirstTickHasACasualty()
    task = asyncio.create_task(run_loop(runner, scheduler))
    await asyncio.sleep(0.05)
    task.cancel()

    assert [c.is_prefill for c in runner.calls[:2]] == [True, True]


@pytest.mark.anyio
async def test_ragged_batch_runs_to_completion():
    runner = FakeRunner(script=lambda call, row: (row + 1) * 3 + call)
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    short = seq(2, max_new_tokens=2, eos_token_ids={99})
    long = seq(6, max_new_tokens=3, eos_token_ids={99})
    scheduler.add(short)
    scheduler.add(long)

    task = asyncio.create_task(run_loop(runner, scheduler))
    await asyncio.sleep(0.1)
    task.cancel()

    assert (short.finish_reason, long.finish_reason) == ("length", "length")
    assert (len(short.output_ids), len(long.output_ids)) == (2, 3)


def test_prompt_to_sequence_takes_tokenization_from_the_runner():
    """The scheduler never sees a tokenizer; the runner owns the vocab."""
    runner = FakeRunner()
    runner.eos_token_ids = {42}

    s = prompt_to_sequence(runner, "hi", 8, 0.5, 20)

    assert s.prompt_ids == [ord("h"), ord("i")]
    assert s.eos_token_ids == {42}
    assert (s.max_new_tokens, s.temperature, s.top_k) == (8, 0.5, 20)
