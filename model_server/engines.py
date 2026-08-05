"""The engine: a per-request view onto one shared scheduling loop.

    main.py      HTTP only
    engines.py   BatchingEngine -- this file
    batching.py  Scheduler + run_loop (WHAT runs, no torch)
    runners.py   ModelRunner (HOW it runs)

There is exactly one engine now. Milestones 2 and 3 had three -- MockEngine,
NanoGPTEngine, QwenEngine -- each driving its own generation and letting main.py's
semaphore do the concurrency accounting. That worked while every request was
generated independently, and stopped working the moment a scheduler had to see all
of them at once. Rather than keep both regimes and a flag to pick between them,
the mock moved onto the same scheduler (see runners.MockRunner) and the semaphore
was deleted outright.

What that bought: main.py holds no concurrency logic at all, `vllm:num_requests_*`
has a single source, and the mock image measures the real admission rule instead
of simulating one.
"""

from dataclasses import dataclass


@dataclass
class Completion:
    """What the engine returns: text plus honest token counts. The engine owns
    tokenization, so it is the only thing that can count truthfully -- the HTTP
    layer just copies these into the usage block."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "length"


class BatchingEngine:
    """Submits a Sequence, then reads that sequence's queue. That is all.

    It generates nothing itself: every forward pass happens on the single
    background run_loop shared by all requests. Fan-out from one batched forward
    back to N independent HTTP responses is entirely `Sequence.queue`, which is
    why the HTTP layer never learns that batching exists.
    """

    def __init__(self, runner, scheduler):
        self.runner = runner
        self.scheduler = scheduler

    def stats(self) -> tuple[int, int]:
        """(running, waiting) for /metrics -- a measurement, not a simulation.

        `waiting` is now "could not start because the batch is full", not "queued
        behind a number I made up". Same metric name, same Grafana panel, first
        time it means something.
        """
        return self.scheduler.num_running, self.scheduler.num_waiting

    def _submit(self, prompt, max_tokens, temperature, top_k):
        from batching import prompt_to_sequence

        seq = prompt_to_sequence(self.runner, prompt, max_tokens, temperature, top_k)
        self.scheduler.add(seq)
        return seq

    def cancel(self, seq) -> None:
        """Abandon a sequence whose client went away. No-op if it already finished."""
        self.scheduler.abort(seq)

    async def generate(self, prompt, max_tokens, temperature, top_k) -> Completion:
        seq = self._submit(prompt, max_tokens, temperature, top_k)
        try:
            # Wait for the sentinel, not for a token count: only the scheduler
            # knows whether this stopped on EOS, on max_new_tokens, or on an abort.
            # The tokens themselves are already in seq.output_ids -- for a one-shot
            # request the queue is purely a wakeup channel.
            while await seq.queue.get() is not None:
                pass
        finally:
            # Not optional. On client disconnect FastAPI cancels this coroutine and
            # CancelledError lands on the await above; without this the sequence
            # keeps its batch slot and generates to max_new_tokens for nobody.
            # cancel() is a no-op on a normally-finished sequence, so one
            # unconditional call covers both exits.
            self.cancel(seq)

        return Completion(
            text=self.runner.decode(seq.output_ids),
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids),
            finish_reason=seq.finish_reason,
        )

    async def stream(self, prompt, max_tokens, temperature, top_k):
        seq = self._submit(prompt, max_tokens, temperature, top_k)
        sent = 0
        try:
            while await seq.queue.get() is not None:
                # Decode the WHOLE output each time and yield only the new suffix.
                # One CJK character or emoji is routinely 2-3 BPE tokens, so
                # decoding each token alone produces U+FFFD where they should be.
                # An incomplete character contributes no characters, so it simply
                # produces no yield that round and appears once complete.
                text = self.runner.decode(seq.output_ids)
                if len(text) > sent:
                    yield text[sent:]
                    sent = len(text)
        finally:
            # Streaming clients are the ones that hang up early.
            self.cancel(seq)
