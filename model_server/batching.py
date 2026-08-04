"""Continuous batching: one scheduler loop serving many concurrent requests.

Milestone 4 skeleton (see docs/continuous-batching-tutorial.md). The plumbing --
Sequence bookkeeping, the async wrapper, the engine adapter -- is given. The six
TODOs are the parts worth learning: left padding, admission control, per-sequence
sampling, finish detection, the manual forward pass, and the tick that ties them
together. `pytest test_batching.py` is the grader.

The shape of the whole thing, and why it is shaped that way:

    main.py          one asyncio.Queue per request, no semaphore
    engines.py       BatchingEngine -- adapts this loop to the Engine ABC
    batching.py      Scheduler + step()  <- this file, owns "when does what run"
    transformers     raw forward passes, no generate()

Milestones 2 and 3 could stay behind `Engine.generate(prompt, ...)` because each
request was generated independently. Continuous batching breaks that assumption:
the scheduler has to see every in-flight request at once, so concurrency
accounting moves OUT of main.py's semaphore and INTO this file. That is the
architectural point of the milestone, not an incidental refactor.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from itertools import count

import torch

# Monotonic ids. Sequences are compared and de-duplicated by id, never by object
# identity, because the scheduler moves them between deques.
_next_id = count()


def _running_loop():
    """The loop a Sequence was created on, or None if it was created synchronously."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


@dataclass
class Sequence:
    """One in-flight request: its tokens, its sampling params, its output channel.

    `queue` is what makes streaming work without the HTTP layer knowing anything
    about batching. The loop pushes each sampled token into the queue of the
    sequence it belongs to; the request handler awaits its own queue. Fan-out from
    one shared batch back to N independent responses is exactly this one field.
    """

    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.7
    top_k: int | None = 20
    eos_token_ids: set[int] = field(default_factory=set)
    id: int = field(default_factory=lambda: next(_next_id))
    output_ids: list[int] = field(default_factory=list)
    # "" while running; "stop" (hit EOS) or "length" (hit max_new_tokens) when done.
    finish_reason: str = ""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = field(default_factory=_running_loop)

    def emit(self, token: int | None) -> None:
        """Hand one token to whoever is streaming this sequence. Thread-safe.

        This indirection is not decoration. step() runs inside asyncio.to_thread,
        so postprocess executes on a WORKER thread -- and asyncio.Queue is not
        thread-safe. Calling put_nowait directly from the worker races with the
        consumer's `await queue.get()`: the queue wakes its waiter by touching a
        Future, and Futures may only be touched from their own loop. The failure
        is rare, load-dependent, and shows up as a stream that hangs forever --
        the worst possible bug to debug. call_soon_threadsafe is the supported
        way to cross that boundary.

        `loop` is None when the Sequence was built outside a running loop (the
        grader does this), in which case a plain put_nowait is correct.
        """
        if self.loop is None:
            self.queue.put_nowait(token)
        else:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, token)

    @property
    def token_ids(self) -> list[int]:
        """Everything the model has seen for this sequence: prompt + what it wrote."""
        return self.prompt_ids + self.output_ids

    @property
    def is_finished(self) -> bool:
        return bool(self.finish_reason)

    def __len__(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


# ---------------------------------------------------------------------------
# TODO 1 -- left padding
# ---------------------------------------------------------------------------


def pad_and_stack(
    seqs: list[Sequence], pad_id: int, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack ragged sequences into one rectangular batch, padding on the LEFT.

    Returns (input_ids, attention_mask, position_ids), each (B, T) int64, where
    T = max(len(s) for s in seqs).

    Three things this has to get right:

    - LEFT padding, not right. Decode always reads logits at the last column. With
      right padding the last column of a short row is a pad token, so you would
      sample the continuation of <pad>. Left padding puts every row's real last
      token at index -1, which is what makes a single [:, -1, :] slice correct for
      the whole batch.
    - attention_mask is 0 on pads, 1 on real tokens. Without it the model attends
      to padding and the short rows silently produce garbage.
    - position_ids must SKIP the pads. RoPE positions come from this tensor, so a
      row padded by k must start at position 0 at its first real token, not at k.
      Pad positions themselves are never attended to, so any value works there --
      clamp to 0 rather than letting them go negative.

    Getting position_ids wrong is the nastiest bug in this milestone: nothing
    crashes, batch=1 is unaffected (no padding), and the output only degrades once
    sequences of different lengths share a batch.
    """
    width = max(len(s) for s  in seqs)
    ids, masks = [], []

    for s in seqs:
        toks = s.token_ids
        pad = width - len(toks)
        ids.append([pad_id] * pad + toks)
        masks.append([0] * pad + [1] * len(toks))

    input_ids = torch.tensor(ids, dtype=torch.int64, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.int64, device=device)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids



# ---------------------------------------------------------------------------
# TODO 2 + TODO 4 -- the scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Two deques and an admission rule. That is the whole thing.

    `waiting` is requests that have arrived but hold no KV cache. `running` is
    requests currently occupying cache. The split is what turns MAX_CONCURRENCY
    from an arbitrary number in a config file into a real capacity limit -- and
    what makes `vllm:num_requests_waiting` a measurement rather than a simulation.
    """

    def __init__(self, max_batch_size: int = 8, max_batch_tokens: int = 8192):
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens

    def add(self, seq: Sequence) -> None:
        """Enqueue a new request. Always accepted -- backpressure is the queue."""
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """Admit what fits, then return the full running set for this tick.

        Rules, in order:

        1. Drop finished sequences out of `running` first. Their cache is freed,
           which is what makes room for the next admission -- a finished sequence
           still occupying a slot is the bug that makes throughput collapse to
           static batching.
        2. Admit from the FRONT of `waiting` while BOTH limits hold:
             len(running) + 1 <= max_batch_size
             total tokens across running + the candidate <= max_batch_tokens
           Use len(seq) for the candidate's cost (prompt + whatever it has
           already generated -- normally just the prompt at admission time).
        3. One exception, and it is not optional: if `running` is EMPTY, admit the
           head unconditionally, even if it exceeds max_batch_tokens on its own.
           Without this a prompt longer than the token budget can never be
           admitted -- it sits at the front of the queue forever, blocking
           everything behind it, and the server looks hung rather than busy.
           Admitting it may exceed the budget; refusing it deadlocks. Real
           schedulers make the same trade.
        4. Stop at the first candidate that does not fit. Do NOT skip it and try
           the next one: FIFO is what bounds tail latency. Best-fit packing looks
           smarter on a throughput graph and starves long prompts forever.
        5. Return list(self.running).

        The returned list is the full running set, not just the newly admitted
        ones. step() re-prefills the whole batch whenever membership changes --
        see the note there for why, and what it costs.
        """
        # Drop finished sequence out of running queue
        self.running = deque(s for s in self.running if not s.is_finished)

        while self.waiting:
            candidate = self.waiting[0]
            if len(self.running) + 1 > self.max_batch_size:
                break

            total = sum(len(s) for s in self.running) + len(candidate)

            if self.running and total > self.max_batch_tokens:
                break

            self.running.append(self.waiting.popleft())

        return list(self.running)


    def postprocess(self, batch: list[Sequence], next_ids: list[int]) -> list[Sequence]:
        """Append one sampled token per sequence and decide who is done.

        `next_ids[i]` belongs to `batch[i]`. For each pair:

        - append the token to seq.output_ids
        - hand it to the streaming consumer with seq.emit(token) -- never
          seq.queue.put_nowait directly; see Sequence.emit for why
        - if the token is seq.eos_token_id -> finish_reason = "stop"
          elif len(seq.output_ids) >= seq.max_new_tokens -> finish_reason = "length"
        - when a sequence finishes, seq.emit(None). The consumer needs an explicit
          end-of-stream marker; an empty queue is indistinguishable from "the next
          token is still being computed".

        Order matters for EOS: check EOS *before* the length limit, so a sequence
        that emits EOS on exactly its last allowed token reports "stop" and not
        "length". The grader checks this precise case.

        Do NOT append the EOS token to output_ids... actually, do append it -- the
        engine strips special tokens at decode time, and keeping it makes the
        token count honest. What you must not do is push it and then also mark
        the sequence unfinished.

        Returns the sequences that finished on this tick (possibly empty).
        """
        finished = []

        for s, token in zip(batch, next_ids):
            s.output_ids.append(token)
            s.emit(token)

            if token in s.eos_token_ids:
                s.finish_reason = "stop"
            elif len(s.output_ids) >= s.max_new_tokens:
                s.finish_reason = "length"

            if s.is_finished:
                s.emit(None)
                finished.append(s)

        return finished


    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)


# ---------------------------------------------------------------------------
# TODO 3 -- per-sequence sampling
# ---------------------------------------------------------------------------


def sample_tokens(logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
    """Sample one token per row, honouring each row's OWN temperature and top_k.

    logits: (B, vocab) -- already sliced to the last position by forward_batch.

    This is where batching stops being free. `model.generate()` applies one
    sampling config to the whole call, but here row 0 may be temperature=0 while
    row 1 is temperature=0.9 with top_k=20, because they are different HTTP
    requests that happened to share a forward pass. A single vectorised softmax
    over the batch would silently apply one row's params to all of them.

    Per row, matching qwen.sampling_kwargs' semantics exactly:
      temperature <= 0        -> greedy: argmax, no sampling
      otherwise               -> divide by temperature, apply top_k if set and > 0,
                                 softmax, then torch.multinomial
    top_k of None / 0 / negative all mean "unset" -- same convention as Milestone 3.

    A per-row Python loop is the right call at these batch sizes: sampling is
    microseconds against a forward pass of milliseconds. Vectorising it is a real
    optimisation only once batches are in the hundreds, and it costs you the
    per-request params. Returns a plain list[int] of length B.
    """
    picked = []

    for row, s in enumerate(seqs):
        row_logits = logits[row]
        if s.temperature <= 0:
            picked.append(int(torch.argmax(row_logits)))
            continue

        scaled = row_logits / s.temperature
        if s.top_k and s.top_k > 0:
            k = min(s.top_k, scaled.size(-1))
            kept, _ = torch.topk(scaled, k)
            scaled = scaled.masked_fill(scaled < kept[-1], float("-inf"))

        probs = torch.softmax(scaled, dim=-1)
        picked.append(int(torch.multinomial(probs, 1)))

    return picked




# ---------------------------------------------------------------------------
# TODO 5 -- the manual forward pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_batch(
    model,
    seqs: list[Sequence],
    pad_id: int,
    cache=None,
) -> tuple[torch.Tensor, object]:
    """One forward pass over the batch. Returns (last-position logits, cache).

    This is the step that leaves `generate()` behind for good. generate() owns its
    own loop and cannot be interrupted to add or drop a sequence, which is
    precisely what continuous batching needs to do -- so we drive the model
    directly and keep the KV cache ourselves.

    Two paths, and they differ ONLY in how much of the input you feed:

    prefill (cache is None):
        feed the whole padded batch. input_ids/attention_mask/position_ids all
        come straight from pad_and_stack.

    decode (cache is not None):
        the cache already holds every token except the newest, so feed only the
        last column: input_ids[:, -1:] and position_ids[:, -1:]. The attention
        mask is the exception -- pass it in FULL, all T columns. It describes what
        may be attended to (the whole history), not what is being fed. Slicing the
        mask to one column is the classic bug here: it makes every sequence attend
        only to its own newest token, so the model forgets everything and starts
        emitting fluent nonsense.

    Call the model with use_cache=True and return
    (out.logits[:, -1, :], out.past_key_values).

    Note there is no torch.compile and no static cache here. Both were measured in
    bench.py; both are batch=1 optimisations whose value shrinks as the batch
    grows, and one of them is blocked on a version bug. See the tutorial appendix.
    """
    device = getattr(model, "device", "cpu")
    input_ids, attention_mask, position_ids = pad_and_stack(seqs, pad_id, device)
    if cache is not None:
        input_ids = input_ids[:, -1:]
        position_ids = position_ids[:, -1:]

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True
    )

    return out.logits[:, -1, :], out.past_key_values



# ---------------------------------------------------------------------------
# TODO 6 -- one tick of the loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def step(model, scheduler: Scheduler, pad_id: int, state: dict) -> list[Sequence]:
    """Advance every running sequence by exactly one token. Returns who finished.

    `state` is the loop's memory between ticks, a dict with two keys:
        state["members"]  tuple[int, ...]  -- seq ids the cache was built for
        state["cache"]                     -- the KV cache, or None

    The tick:

    1. batch = scheduler.schedule(). If it is empty, reset state (members=(),
       cache=None) and return [] -- an idle loop must not hold a stale cache.
    2. Compare tuple(s.id for s in batch) against state["members"]. If they
       differ, the cache no longer describes this batch: set cache = None, which
       forces a prefill of the whole batch on this tick.
    3. logits, cache = forward_batch(model, batch, pad_id, cache)
    4. next_ids = sample_tokens(logits, batch)
    5. finished = scheduler.postprocess(batch, next_ids)
    6. Save cache and members back into state, then return finished.

    Step 2 is the honest compromise in this implementation, and you should know
    exactly what it costs. Rebuilding the cache on every membership change means a
    single request finishing forces a full re-prefill of everyone still running --
    O(batch x context) wasted work at every arrival and every completion. It is
    correct, it is about fifteen lines, and it still beats the semaphore by a wide
    margin because decode ticks vastly outnumber membership changes.

    What it is not is what vLLM does. Paged attention exists to make exactly this
    free: KV cache in fixed-size blocks with a per-sequence block table, so adding
    or dropping a sequence rewrites a table instead of recomputing attention. The
    appendix walks through transformers' own PagedAttentionCache, which you are
    now in a position to read. Implementing it is Milestone 5, if you want it.
    """
    batch = scheduler.schedule()
    if not batch:
        state["members"] = ()
        state["cache"] = None
        return []

    members = tuple(s.id for s in batch)
    cache = state["cache"] if members == state["members"] else None

    logits, cache = forward_batch(model, batch, pad_id, cache)
    next_ids = sample_tokens(logits, batch)
    finished = scheduler.postprocess(batch, next_ids)

    state["members"] = members
    state["cache"] = cache

    return finished


# ---------------------------------------------------------------------------
# Plumbing used by engines.py -- given, nothing to implement below.
# ---------------------------------------------------------------------------


async def run_loop(model, scheduler: Scheduler, pad_id: int, idle_sleep: float = 0.005):
    """Drive step() forever, off the event loop.

    Every tick is a forward pass, so it goes through asyncio.to_thread exactly as
    Milestones 2 and 3 did -- /health and /metrics have to keep answering while
    the GPU is busy. The difference is that there is now ONE loop for the whole
    process instead of one thread per request.

    When there is no work the loop sleeps briefly rather than spinning. A
    condition variable would be tidier; a 5ms poll is easier to read and adds at
    most 5ms to the latency of a request that arrives into an idle server.
    """
    state: dict = {"members": (), "cache": None}
    while True:
        if not scheduler.has_work():
            await asyncio.sleep(idle_sleep)
            continue
        await asyncio.to_thread(step, model, scheduler, pad_id, state)


def resolve_eos_ids(model, tokenizer) -> set[int]:
    """Every token id that means "stop" for this model -- not just the advertised one.

    Qwen3 has TWO: <|im_end|> (151645) and <|endoftext|> (151643). Only the first
    is `tokenizer.eos_token_id`; the full list lives in `generation_config`, which
    is where generate() read it from on your behalf in Milestone 3. Drive the model
    yourself and nobody reads it for you -- so a sequence that ends naturally on
    <|endoftext|> is not recognised, keeps generating until max_new_tokens, trails
    a stretch of degenerate text, and reports "length". That quietly undoes
    Milestone 3's "finish_reason finally tells the truth".

    Resolved ONCE, at engine construction. It is a property of the model, and
    prompt_to_sequence has no business holding a model reference just to compute
    the same set on every request.
    """
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return set()  # no EOS at all (nano-GPT); max_new_tokens is the only stop
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def prompt_to_sequence(tokenizer, prompt: str, max_new_tokens: int,
                       temperature: float, top_k: int | None,
                       eos_token_ids: set[int],
                       enable_thinking: bool = False) -> Sequence:
    """Tokenize an HTTP request into a Sequence the scheduler can accept.

    Reuses Milestone 3's build_prompt so the chat template stays in one place --
    batching changes how sequences are executed, never how they are formatted.

    `eos_token_ids` comes in from the caller rather than being derived here; see
    resolve_eos_ids for why.
    """
    from qwen import build_prompt

    text = build_prompt(tokenizer, prompt, enable_thinking)
    return Sequence(
        prompt_ids=tokenizer(text).input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_ids=eos_token_ids,
    )
