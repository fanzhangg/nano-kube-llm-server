"""Grader for the Milestone 4 continuous-batching skeleton.

See docs/continuous-batching-tutorial.md. These tests define the contract for
every TODO in batching.py -- work through the skeleton until
`pytest test_batching.py -v` is green. Each test name states the property it
checks; when one goes red, read the name before reading the traceback.

Deliberately cheap: no GPU, no weights, not even transformers. The model is a
stub that records the tensors it was handed and returns logits we chose, because
what is under test is the *scheduling contract* -- is padding on the left, are
position ids skipping pads, is the cache invalidated when membership changes,
does a sequence that emits EOS on its last allowed token report "stop" -- and
none of that needs a real forward pass to grade. The one thing a stub cannot
catch is whether real Qwen3 output is coherent; Day 5's end-to-end run covers it.
"""

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="milestone 4 needs torch")

from batching import (  # noqa: E402
    Scheduler,
    Sequence,
    forward_batch,
    pad_and_stack,
    sample_tokens,
    step,
)

VOCAB = 16


def seq(prompt_len: int, max_new_tokens: int = 4, **kw) -> Sequence:
    """A Sequence with a distinct, non-zero prompt so padding is visible."""
    return Sequence(
        prompt_ids=list(range(1, prompt_len + 1)),
        max_new_tokens=max_new_tokens,
        temperature=kw.pop("temperature", 0.0),
        top_k=kw.pop("top_k", None),
        **kw,
    )


class StubModel:
    """Records every call, returns logits whose argmax we control.

    Mimics only the surface batching.py actually touches: called with keyword
    tensors, returns an object with `.logits` (B, T, VOCAB) and
    `.past_key_values`. The returned cache is a plain sentinel -- the grader only
    ever checks whether it was passed back in, never what is inside it.
    """

    def __init__(self, forced=None):
        # forced(call_index, row_index) -> token id that must win the argmax.
        self.forced = forced or (lambda call, row: 7)
        self.calls: list[dict] = []
        self.device = torch.device("cpu")

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=False, **_):
        self.calls.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        })
        call_index = len(self.calls) - 1
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, VOCAB)
        for row in range(batch):
            logits[row, -1, self.forced(call_index, row)] = 100.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=SimpleNamespace(tag=f"cache-{call_index}"),
        )

    @property
    def last(self) -> dict:
        return self.calls[-1]


# ---------------------------------------------------------------------------
# TODO 1 -- pad_and_stack
# ---------------------------------------------------------------------------


@pytest.mark.todo1
def test_pads_on_the_left_not_the_right():
    short, long = seq(3), seq(5)
    input_ids, _, _ = pad_and_stack([short, long], pad_id=0)

    assert input_ids.shape == (2, 5)
    # The short row's real tokens must sit flush against the RIGHT edge, so that
    # logits[:, -1] is every row's genuine last token.
    assert input_ids[0].tolist() == [0, 0, 1, 2, 3]
    assert input_ids[1].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.todo1
def test_attention_mask_is_zero_exactly_on_pads():
    _, mask, _ = pad_and_stack([seq(3), seq(5)], pad_id=0)

    assert mask[0].tolist() == [0, 0, 1, 1, 1]
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


@pytest.mark.todo1
def test_position_ids_skip_the_padding():
    """The bug that only shows up once ragged sequences share a batch.

    RoPE reads positions from this tensor. A row padded by two must still begin
    at position 0 on its first real token -- if it starts at 2, the model sees
    every token shifted, and the output degrades without anything crashing.
    """
    _, _, positions = pad_and_stack([seq(3), seq(5)], pad_id=0)

    assert positions[0].tolist() == [0, 0, 0, 1, 2]
    assert positions[1].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.todo1
def test_uses_a_custom_pad_id():
    input_ids, _, _ = pad_and_stack([seq(2), seq(4)], pad_id=151643)

    assert input_ids[0].tolist()[:2] == [151643, 151643]


@pytest.mark.todo1
def test_single_sequence_needs_no_padding():
    input_ids, mask, positions = pad_and_stack([seq(4)], pad_id=0)

    assert input_ids.tolist() == [[1, 2, 3, 4]]
    assert mask.tolist() == [[1, 1, 1, 1]]
    assert positions.tolist() == [[0, 1, 2, 3]]


@pytest.mark.todo1
def test_includes_generated_tokens_not_just_the_prompt():
    """A running sequence's cache covers prompt + everything it has written."""
    running = seq(2)
    running.output_ids = [9, 9]
    input_ids, _, _ = pad_and_stack([running], pad_id=0)

    assert input_ids.tolist() == [[1, 2, 9, 9]]


@pytest.mark.todo1
def test_returns_int64_tensors():
    """Embedding lookups and RoPE both require integer indices."""
    for tensor in pad_and_stack([seq(3), seq(5)], pad_id=0):
        assert tensor.dtype == torch.int64


# ---------------------------------------------------------------------------
# TODO 2 -- Scheduler.schedule
# ---------------------------------------------------------------------------


@pytest.mark.todo2
def test_admits_waiting_requests_up_to_the_batch_size():
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=1000)
    for _ in range(3):
        scheduler.add(seq(4))

    batch = scheduler.schedule()

    assert len(batch) == 2
    assert scheduler.num_running == 2
    assert scheduler.num_waiting == 1


@pytest.mark.todo2
def test_admits_in_fifo_order():
    scheduler = Scheduler(max_batch_size=1, max_batch_tokens=1000)
    first, second = seq(4), seq(4)
    scheduler.add(first)
    scheduler.add(second)

    assert [s.id for s in scheduler.schedule()] == [first.id]


@pytest.mark.todo2
def test_respects_the_token_budget():
    scheduler = Scheduler(max_batch_size=10, max_batch_tokens=10)
    scheduler.add(seq(6))
    scheduler.add(seq(6))

    batch = scheduler.schedule()

    # 6 fits; 6 + 6 = 12 does not.
    assert len(batch) == 1
    assert scheduler.num_waiting == 1


@pytest.mark.todo2
def test_stops_at_the_first_request_that_does_not_fit():
    """FIFO, not best-fit. Packing the small one in first starves the big one."""
    scheduler = Scheduler(max_batch_size=10, max_batch_tokens=5)
    scheduler.add(seq(4))
    scheduler.schedule()          # the len-4 request is now running
    big, small = seq(9), seq(1)
    scheduler.add(big)
    scheduler.add(small)

    batch = scheduler.schedule()

    assert len(batch) == 1, "a request behind a too-large one must not jump the queue"
    assert [s.id for s in scheduler.waiting] == [big.id, small.id]


@pytest.mark.todo2
def test_admits_an_oversized_prompt_into_an_empty_batch():
    """Refusing it would deadlock: it can never reach the front of an empty queue."""
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=5)
    scheduler.add(seq(50))

    assert len(scheduler.schedule()) == 1


@pytest.mark.todo2
def test_drops_finished_sequences_and_backfills():
    """The whole point of *continuous* batching: a freed slot refills immediately."""
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=1000)
    a, b, c = seq(4), seq(4), seq(4)
    for s in (a, b, c):
        scheduler.add(s)
    scheduler.schedule()
    a.finish_reason = "stop"

    batch = scheduler.schedule()

    ids = {s.id for s in batch}
    assert a.id not in ids, "a finished sequence must release its slot"
    assert c.id in ids, "the freed slot must be backfilled on the same tick"


@pytest.mark.todo2
def test_returns_the_full_running_set_not_only_new_admissions():
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    scheduler.add(seq(4))
    scheduler.schedule()
    scheduler.add(seq(4))

    assert len(scheduler.schedule()) == 2


@pytest.mark.todo2
def test_idle_scheduler_schedules_nothing():
    scheduler = Scheduler()

    assert scheduler.schedule() == []
    assert not scheduler.has_work()


# ---------------------------------------------------------------------------
# TODO 3 -- sample_tokens
# ---------------------------------------------------------------------------


@pytest.mark.todo3
def test_greedy_when_temperature_is_zero():
    logits = torch.zeros(1, VOCAB)
    logits[0, 11] = 5.0

    assert sample_tokens(logits, [seq(2, temperature=0.0)]) == [11]


@pytest.mark.todo3
def test_each_row_uses_its_own_sampling_params():
    """Two HTTP requests sharing one forward pass do NOT share a sampling config.

    Row 0 is greedy and must return its argmax. Row 1 is sampled with top_k=1,
    which collapses to its own argmax -- a different token. One vectorised
    softmax over the batch would apply a single config to both.
    """
    logits = torch.zeros(2, VOCAB)
    logits[0, 3] = 10.0
    logits[1, 12] = 10.0

    picked = sample_tokens(
        logits, [seq(2, temperature=0.0), seq(2, temperature=0.9, top_k=1)]
    )

    assert picked == [3, 12]


@pytest.mark.todo3
def test_top_k_restricts_the_candidate_set():
    """With top_k=1 sampling is deterministic, so this is checkable exactly."""
    logits = torch.zeros(1, VOCAB)
    logits[0, 5] = 1.0
    logits[0, 6] = 0.9

    for _ in range(20):
        assert sample_tokens(logits, [seq(2, temperature=1.0, top_k=1)]) == [5]


@pytest.mark.todo3
def test_unset_top_k_does_not_filter_everything():
    """None / 0 / negative all mean "no filter" -- 0 must not read as "zero candidates"."""
    logits = torch.zeros(1, VOCAB)
    logits[0, 4] = 50.0

    for unset in (None, 0, -1):
        assert sample_tokens(logits, [seq(2, temperature=0.5, top_k=unset)]) == [4]


@pytest.mark.todo3
def test_returns_plain_ints_one_per_row():
    logits = torch.randn(3, VOCAB)

    picked = sample_tokens(logits, [seq(2), seq(2), seq(2)])

    assert len(picked) == 3
    assert all(isinstance(t, int) for t in picked), "return ints, not 0-dim tensors"


# ---------------------------------------------------------------------------
# TODO 4 -- Scheduler.postprocess
# ---------------------------------------------------------------------------


@pytest.mark.todo4
def test_appends_the_sampled_token_to_its_own_sequence():
    scheduler = Scheduler()
    a, b = seq(2), seq(2)

    scheduler.postprocess([a, b], [5, 6])

    assert a.output_ids == [5]
    assert b.output_ids == [6]


@pytest.mark.todo4
def test_pushes_each_token_into_its_own_queue():
    """Fan-out from one shared batch back to N independent HTTP responses."""
    scheduler = Scheduler()
    a, b = seq(2), seq(2)

    scheduler.postprocess([a, b], [5, 6])

    assert a.queue.get_nowait() == 5
    assert b.queue.get_nowait() == 6


@pytest.mark.todo4
def test_eos_finishes_the_sequence_with_stop():
    scheduler = Scheduler()
    s = seq(2, max_new_tokens=10, eos_token_ids={2})

    finished = scheduler.postprocess([s], [2])

    assert s.finish_reason == "stop"
    assert [f.id for f in finished] == [s.id]


@pytest.mark.todo4
def test_hitting_max_new_tokens_finishes_with_length():
    scheduler = Scheduler()
    s = seq(2, max_new_tokens=2, eos_token_ids={99})

    assert scheduler.postprocess([s], [5]) == []
    assert s.finish_reason == ""

    finished = scheduler.postprocess([s], [6])

    assert s.finish_reason == "length"
    assert [f.id for f in finished] == [s.id]


@pytest.mark.todo4
def test_eos_on_the_final_allowed_token_reports_stop_not_length():
    """Check EOS before the length limit, or a clean stop is reported as truncation."""
    scheduler = Scheduler()
    s = seq(2, max_new_tokens=1, eos_token_ids={2})

    scheduler.postprocess([s], [2])

    assert s.finish_reason == "stop"


@pytest.mark.todo4
def test_finished_sequence_gets_an_end_of_stream_sentinel():
    """An empty queue means "not yet"; None means "never again"."""
    scheduler = Scheduler()
    s = seq(2, max_new_tokens=1, eos_token_ids={99})

    scheduler.postprocess([s], [5])

    assert s.queue.get_nowait() == 5
    assert s.queue.get_nowait() is None


@pytest.mark.todo4
def test_unfinished_sequence_gets_no_sentinel():
    scheduler = Scheduler()
    s = seq(2, max_new_tokens=5, eos_token_ids={99})

    scheduler.postprocess([s], [5])
    s.queue.get_nowait()

    assert s.queue.empty()


@pytest.mark.todo4
def test_one_sequence_finishing_does_not_disturb_the_others():
    scheduler = Scheduler()
    done = seq(2, max_new_tokens=1, eos_token_ids={99})
    alive = seq(2, max_new_tokens=9, eos_token_ids={99})

    finished = scheduler.postprocess([done, alive], [1, 2])

    assert [f.id for f in finished] == [done.id]
    assert alive.finish_reason == ""


# ---------------------------------------------------------------------------
# TODO 5 -- forward_batch
# ---------------------------------------------------------------------------


@pytest.mark.todo5
def test_prefill_feeds_the_whole_padded_batch():
    model = StubModel()
    batch = [seq(3), seq(5)]

    forward_batch(model, batch, pad_id=0, cache=None)

    assert model.last["input_ids"].shape == (2, 5)
    assert model.last["past_key_values"] is None
    assert model.last["use_cache"] is True


@pytest.mark.todo5
def test_decode_feeds_only_the_newest_column():
    model = StubModel()
    running = seq(3)
    running.output_ids = [9]

    forward_batch(model, [running], pad_id=0, cache=SimpleNamespace(tag="warm"))

    assert model.last["input_ids"].shape == (1, 1)
    assert model.last["input_ids"].tolist() == [[9]], "feed the last token, not the first"
    assert model.last["position_ids"].tolist() == [[3]]


@pytest.mark.todo5
def test_decode_passes_the_attention_mask_in_full():
    """The mask describes what may be ATTENDED TO, not what is being fed.

    Slicing it to one column alongside input_ids makes every sequence attend only
    to its own newest token: the model forgets its context and emits fluent
    nonsense, with nothing raising.
    """
    model = StubModel()
    running = seq(3)
    running.output_ids = [9]

    forward_batch(model, [running], pad_id=0, cache=SimpleNamespace(tag="warm"))

    assert model.last["attention_mask"].shape == (1, 4)


@pytest.mark.todo5
def test_decode_reuses_the_cache_it_was_given():
    model = StubModel()
    warm = SimpleNamespace(tag="warm")

    forward_batch(model, [seq(3)], pad_id=0, cache=warm)

    assert model.last["past_key_values"] is warm


@pytest.mark.todo5
def test_returns_last_position_logits_and_the_new_cache():
    model = StubModel()

    logits, cache = forward_batch(model, [seq(3), seq(5)], pad_id=0, cache=None)

    assert logits.shape == (2, VOCAB), "slice to the last position before sampling"
    assert cache is not None and cache.tag == "cache-0"


# ---------------------------------------------------------------------------
# TODO 6 -- step
# ---------------------------------------------------------------------------


def fresh_state() -> dict:
    return {"members": (), "cache": None}


@pytest.mark.todo6
def test_first_tick_prefills():
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    scheduler.add(seq(3))

    step(model, scheduler, pad_id=0, state=state)

    assert model.calls[0]["past_key_values"] is None


@pytest.mark.todo6
def test_second_tick_decodes_with_the_cache():
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    scheduler.add(seq(3, max_new_tokens=9, eos_token_ids={99}))

    step(model, scheduler, pad_id=0, state=state)
    step(model, scheduler, pad_id=0, state=state)

    assert model.calls[1]["past_key_values"] is not None
    assert model.calls[1]["input_ids"].shape == (1, 1)


@pytest.mark.todo6
def test_membership_change_forces_a_reprefill():
    """The cache describes a specific set of rows. Change the set, drop the cache.

    Keeping it would feed the new sequence someone else's history -- silently, and
    with plausible-looking output.
    """
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    scheduler.add(seq(3, max_new_tokens=9, eos_token_ids={99}))
    step(model, scheduler, pad_id=0, state=state)
    step(model, scheduler, pad_id=0, state=state)

    scheduler.add(seq(4, max_new_tokens=9, eos_token_ids={99}))
    step(model, scheduler, pad_id=0, state=state)

    assert model.calls[2]["past_key_values"] is None
    assert model.calls[2]["input_ids"].shape[0] == 2


@pytest.mark.todo6
def test_departure_also_forces_a_reprefill():
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    leaving = seq(3, max_new_tokens=1, eos_token_ids={99})
    staying = seq(3, max_new_tokens=9, eos_token_ids={99})
    scheduler.add(leaving)
    scheduler.add(staying)

    step(model, scheduler, pad_id=0, state=state)   # prefill; `leaving` hits its limit
    step(model, scheduler, pad_id=0, state=state)   # membership shrank -> prefill again

    assert leaving.finish_reason == "length"
    assert model.calls[1]["past_key_values"] is None
    assert model.calls[1]["input_ids"].shape[0] == 1


@pytest.mark.todo6
def test_idle_tick_clears_stale_state():
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    scheduler.add(seq(3, max_new_tokens=1, eos_token_ids={99}))
    step(model, scheduler, pad_id=0, state=state)   # finishes immediately

    assert step(model, scheduler, pad_id=0, state=state) == []
    assert state["cache"] is None
    assert state["members"] == ()


@pytest.mark.todo6
def test_step_returns_the_sequences_that_finished():
    model, scheduler, state = StubModel(), Scheduler(), fresh_state()
    s = seq(3, max_new_tokens=1, eos_token_ids={99})
    scheduler.add(s)

    assert [f.id for f in step(model, scheduler, pad_id=0, state=state)] == [s.id]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.todoe2e
def test_ragged_batch_runs_to_completion_with_correct_per_sequence_output():
    """Different prompt lengths, different limits, one shared loop.

    The stub forces row r of call c to emit token (r + 1) * 10 + c, so each
    sequence's output is a unique, checkable fingerprint. If padding, membership
    tracking or the queue fan-out is wrong, the rows bleed into each other here.
    """
    model = StubModel(forced=lambda call, row: (row + 1) * 3 + call)
    scheduler = Scheduler(max_batch_size=4, max_batch_tokens=1000)
    short = seq(2, max_new_tokens=2, eos_token_ids={99})
    long = seq(6, max_new_tokens=3, eos_token_ids={99})
    scheduler.add(short)
    scheduler.add(long)
    state = fresh_state()

    for _ in range(10):
        if not scheduler.has_work():
            break
        step(model, scheduler, pad_id=0, state=state)

    assert short.finish_reason == "length"
    assert long.finish_reason == "length"
    assert len(short.output_ids) == 2
    assert len(long.output_ids) == 3
    assert scheduler.num_running == 0
    assert scheduler.num_waiting == 0


@pytest.mark.todoe2e
def test_queued_request_starts_only_after_a_slot_frees():
    """MAX_CONCURRENCY becomes a real capacity limit, not a simulated one.

    This is the property the whole milestone exists for: while the batch is full
    the third request genuinely waits, so `vllm:num_requests_waiting` reports a
    measurement rather than a number the server made up.
    """
    model = StubModel()
    scheduler = Scheduler(max_batch_size=2, max_batch_tokens=1000)
    a = seq(3, max_new_tokens=1, eos_token_ids={99})
    b = seq(3, max_new_tokens=5, eos_token_ids={99})
    c = seq(3, max_new_tokens=5, eos_token_ids={99})
    for s in (a, b, c):
        scheduler.add(s)
    state = fresh_state()

    step(model, scheduler, pad_id=0, state=state)

    assert scheduler.num_waiting == 1
    assert c.output_ids == [], "must not start before a slot frees"

    step(model, scheduler, pad_id=0, state=state)

    assert scheduler.num_waiting == 0
    assert len(c.output_ids) == 1


@pytest.mark.todoe2e
@pytest.mark.anyio
async def test_streaming_consumer_sees_tokens_then_the_sentinel():
    """What the SSE handler will actually do: await one queue, ignore the batch."""
    model = StubModel(forced=lambda call, row: 5 + call)
    scheduler = Scheduler()
    s = seq(3, max_new_tokens=3, eos_token_ids={99})
    scheduler.add(s)
    state = fresh_state()

    async def drive():
        for _ in range(3):
            await asyncio.to_thread(step, model, scheduler, 0, state)

    task = asyncio.create_task(drive())
    received = []
    while True:
        token = await asyncio.wait_for(s.queue.get(), timeout=5)
        if token is None:
            break
        received.append(token)
    await task

    assert received == [5, 6, 7]
    assert s.finish_reason == "length"
