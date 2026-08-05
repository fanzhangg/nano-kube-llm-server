"""Continuous batching: one scheduler loop serving many concurrent requests.

This module decides **what runs**. It never decides how -- that is runners.py.
The split is the one vLLM and SGLang both settle on:

    main.py      HTTP only: routing, schema, SSE, lifespan
    engines.py   BatchingEngine -- a per-request view onto the shared loop
    batching.py  Sequence + Scheduler + run_loop   <- this file, WHAT runs
    runners.py   ModelRunner: Qwen3 or the mock    <- HOW it runs

**This file imports no torch, and names no model.** That is load-bearing, not
tidiness: it is what lets MockRunner drive the real Scheduler inside the ~150MB
torch-free mock image, so the Kubernetes story -- Pending/Loading/Ready, readiness
gating, queue-depth metrics -- exercises the same scheduling code the GPU image
runs, rather than a simulation of it.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from itertools import count

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
    # "" while running; "stop" (EOS), "length" (max_new_tokens), "cancelled", "error".
    finish_reason: str = ""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = field(default_factory=_running_loop)

    def emit(self, token: int | None) -> None:
        """Hand one token to whoever is streaming this sequence. Thread-safe.

        This indirection is not decoration. The runner executes inside
        asyncio.to_thread, so postprocess runs on a WORKER thread -- and
        asyncio.Queue is not thread-safe. Calling put_nowait directly from the
        worker races with the consumer's `await queue.get()`: the queue wakes its
        waiter by touching a Future, and Futures may only be touched from their
        own loop. The failure is rare, load-dependent, and shows up as a stream
        that hangs forever. call_soon_threadsafe is the supported crossing.

        `loop` is None when the Sequence was built outside a running loop (the
        graders do this), in which case a plain put_nowait is correct.
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


class Scheduler:
    """Two deques and an admission rule. That is the whole thing.

    `waiting` is requests that have arrived but hold no KV cache. `running` is
    requests currently occupying cache. The split is what turns MAX_BATCH_SIZE
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
        # max_tokens=0 is a legal OpenAI request meaning "generate nothing". It has
        # to be resolved HERE: postprocess only ever runs after a token has already
        # been produced, so a zero-budget sequence admitted normally would come back
        # with exactly one token and report "length" for it.
        if seq.max_new_tokens <= 0:
            seq.finish_reason = "length"
            seq.emit(None)
            return
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """Admit what fits, then return the full running set for this tick."""
        # Drop finished sequences FIRST: a completed request still holding a slot
        # is the bug that collapses continuous batching back into static batching.
        self.running = deque(s for s in self.running if not s.is_finished)

        while self.waiting:
            candidate = self.waiting[0]
            if len(self.running) + 1 > self.max_batch_size:
                break

            total = sum(len(s) for s in self.running) + len(candidate)
            # `self.running and`: with an empty batch the head is admitted
            # unconditionally. A prompt longer than the whole token budget would
            # otherwise sit at the front forever, blocking everyone behind it --
            # over budget beats deadlocked, and real schedulers make the same trade.
            if self.running and total > self.max_batch_tokens:
                break

            # break, not skip-ahead: FIFO is what bounds tail latency. Packing a
            # smaller request in first looks better on a throughput graph and
            # starves long prompts forever.
            self.running.append(self.waiting.popleft())

        return list(self.running)

    def postprocess(self, batch: list[Sequence], next_ids: list[int]) -> list[Sequence]:
        """Append one sampled token per sequence and decide who is done."""
        finished = []

        for s, token in zip(batch, next_ids):
            s.output_ids.append(token)
            s.emit(token)

            # EOS before the length limit: a sequence that emits EOS on exactly its
            # last allowed token stopped on its own, and must not report "length".
            if token in s.eos_token_ids:
                s.finish_reason = "stop"
            elif len(s.output_ids) >= s.max_new_tokens:
                s.finish_reason = "length"

            if s.is_finished:
                # The sentinel is emitted AFTER finish_reason is set, which is what
                # lets a consumer that saw None trust finish_reason and output_ids.
                s.emit(None)
                finished.append(s)

        return finished

    @property
    def num_running(self) -> int:
        # Finished sequences stay in the deque until the next schedule() sweeps
        # them, so counting the deque would report a cancelled request as running
        # for up to one tick. /metrics is scraped between ticks more often than
        # not, which would make that staleness the common case rather than a rare
        # one -- and "running" is exactly the number an autoscaler acts on.
        return sum(1 for s in self.running if not s.is_finished)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, seq: Sequence, reason: str = "cancelled") -> None:
        """Give up on one sequence, wherever it currently is.

        No-op on an already-finished sequence, because the engine's `finally`
        calls this unconditionally on the happy path too.
        """
        if seq.is_finished:
            return
        seq.finish_reason = reason
        # schedule() only filters `running`. A cancelled request left in `waiting`
        # would still be admitted later and generate for a client long gone.
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass
        seq.emit(None)

    def fail_all(self, exc: BaseException) -> list[Sequence]:
        """Tear down every in-flight request after the loop hits a fatal error.

        Without this, a dead loop leaves every request blocked on queue.get()
        forever: no error, no 500, no timeout -- a server that accepts work and
        answers none of it. HuggingFace ships the same method (fail_all_requests).
        """
        failed = list(self.running) + list(self.waiting)
        for s in failed:
            if not s.is_finished:
                s.finish_reason = "error"
            s.emit(None)

        self.running.clear()
        self.waiting.clear()
        return failed


async def run_loop(runner, scheduler: Scheduler, idle_sleep: float = 0.005):
    """Drive the runner forever, off the event loop.

    The tick is now three lines, and every model-specific decision has moved
    behind `runner.execute`:

        batch    = scheduler.schedule()          # WHAT runs
        next_ids = runner.execute(ForwardBatch)  # HOW it runs
        scheduler.postprocess(batch, next_ids)   # record what happened

    `is_prefill` is computed HERE and handed to the runner, rather than the runner
    inferring it from "is my cache None". Membership change is scheduling
    information, so the scheduling side owns it -- this is SGLang's ForwardMode,
    and making it explicit is what would let chunked prefill in later.

    A tick that raises takes every in-flight request down with it and re-raises,
    so the task dies loudly. Swallowing the error sounds more robust and is worse:
    after a CUDA OOM the allocator is in an unknown state, and a server that keeps
    accepting work there produces wrong output instead of an outage. Fail the
    batch, kill the loop, let Kubernetes restart the pod.
    """
    from runners import ForwardBatch

    members: tuple[int, ...] = ()
    while True:
        if not scheduler.has_work():
            await asyncio.sleep(idle_sleep)
            continue

        batch = scheduler.schedule()
        if not batch:
            members = ()
            runner.reset()
            continue

        # Order matters, not just membership: the runner's cache is indexed by ROW
        # POSITION, so the same sequences in a different order still invalidate it.
        current = tuple(s.id for s in batch)
        is_prefill = current != members

        try:
            next_ids = await asyncio.to_thread(
                runner.execute, ForwardBatch(seqs=batch, is_prefill=is_prefill)
            )
        except asyncio.CancelledError:
            scheduler.fail_all(asyncio.CancelledError())
            raise
        except BaseException as exc:
            scheduler.fail_all(exc)
            raise

        scheduler.postprocess(batch, next_ids)
        members = current


def prompt_to_sequence(runner, prompt: str, max_new_tokens: int,
                       temperature: float, top_k: int | None) -> Sequence:
    """Turn an HTTP request into a Sequence the scheduler can accept.

    Tokenization is the runner's job -- it owns the chat template and the vocab --
    so this is only wiring. Note the scheduler still never sees a tokenizer.
    """
    return Sequence(
        prompt_ids=runner.encode(prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_ids=runner.eos_token_ids,
    )
