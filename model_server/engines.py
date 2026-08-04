"""Engine implementations: adapters from an inference backend to the HTTP layer.

Three layers, each depending only on the one below it:

    main.py      HTTP: routing, lifecycle, concurrency accounting, metrics
    engines.py   adapters -- this file
    nanogpt.py / qwen.py   inference backends (own the model, know no HTTP)

Nothing here imports FastAPI, and the backend imports are function-local on
purpose: MockEngine must stay usable in environments without torch installed,
which is what keeps test_main.py fast and the mock image small.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Completion:
    """What an Engine returns: the generated text plus honest token counts. The
    engine owns tokenization, so it's the only thing that can count truthfully --
    the HTTP layer just copies these into the usage block."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "length"


class Engine(ABC):
    """The inference boundary. HTTP handlers depend on THIS, never on torch or
    nanogpt directly. Swapping nano-GPT for vLLM later means writing a new Engine
    implementation, not touching a single handler.

    Milestone 4 amended the original contract. It used to read: "the server owns
    concurrency accounting (the _slot semaphore); the engine owns only how to
    generate." That held for two milestones and then broke, because a continuous-
    batching scheduler has to see every in-flight request at once -- accounting
    and generation stop being separable. Rather than let main.py branch on the
    engine's concrete type, the thing that varies is named here:

        owns_concurrency = False  -> main.py's semaphore counts (Mock/NanoGPT/Qwen)
        owns_concurrency = True   -> the engine counts, via stats() (Batching)

    That is the whole seam. It is deliberately small: this project's subject is
    the control plane, and a full vLLM-style EngineCore/frontend split would be
    optimising the scaffolding.
    """

    #: Does this engine do its own admission control? See the class docstring.
    owns_concurrency: bool = False

    @abstractmethod
    async def generate(
        self, prompt: str, max_tokens: int, temperature: float, top_k: int | None
    ) -> Completion:
        """One-shot completion."""

    @abstractmethod
    def stream(
        self, prompt: str, max_tokens: int, temperature: float, top_k: int | None
    ) -> AsyncIterator[str]:
        """Async generator yielding decoded text pieces, one decode step at a time."""

    def stats(self) -> tuple[int, int]:
        """(running, waiting). Only engines with owns_concurrency need this."""
        raise NotImplementedError("this engine does not own concurrency accounting")


class MockEngine(Engine):
    """Fakes prefill+decode latency and returns placeholder text. Used whenever no
    checkpoint is configured, so the server still behaves exactly as the Milestone 1
    mock -- existing tests and old images keep working (graceful degradation)."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def _latency(prompt: str, max_tokens: int) -> float:
        return len(prompt) * 0.001 + max_tokens * 0.02

    @staticmethod
    def _text(max_tokens: int) -> str:
        return f"[mock output of {max_tokens} tokens]"

    async def generate(self, prompt, max_tokens, temperature, top_k) -> Completion:
        await asyncio.sleep(self._latency(prompt, max_tokens))
        return Completion(
            text=self._text(max_tokens),
            prompt_tokens=len(prompt),
            completion_tokens=max_tokens,
        )

    async def stream(self, prompt, max_tokens, temperature, top_k):
        latency = self._latency(prompt, max_tokens)
        pieces = self._text(max_tokens).split(" ")
        per_piece = latency / max(len(pieces), 1)
        for piece in pieces:
            await asyncio.sleep(per_piece)
            yield piece + " "


async def _aiter_blocking(gen):
    """Bridge a blocking generator into an async one, one item per thread hop.

    Every next() on a real engine's stream is one forward pass, so it must not run
    on the event loop. The sentinel marks exhaustion -- returning it from next()
    avoids raising StopIteration across a coroutine boundary, which asyncio mangles
    into a RuntimeError.

    This is engine-agnostic on purpose: nano-GPT and Qwen3 both expose a blocking
    generator, so the bridge is identical for both and belongs here rather than
    copy-pasted into each Engine.
    """
    done = object()
    while True:
        piece = await asyncio.to_thread(next, gen, done)
        if not isinstance(piece, str):  # the sentinel -> generator is exhausted
            break
        yield piece


class NanoGPTEngine(Engine):
    """Real inference over a trained nano-GPT checkpoint. All torch/nanogpt
    dependencies live behind this boundary. CPU/GPU-bound generation is pushed off
    the event loop with asyncio.to_thread so /health and /metrics stay responsive
    while a completion is running."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    async def generate(self, prompt, max_tokens, temperature, top_k) -> Completion:
        from nanogpt import complete

        text = await asyncio.to_thread(
            complete, self.model, self.tokenizer, prompt, max_tokens, temperature, top_k
        )
        # Token counts come from the tokenizer, not from len(text) or max_tokens --
        # this is the usage block finally telling the truth.
        return Completion(
            text=text,
            prompt_tokens=len(self.tokenizer.encode(prompt)),
            completion_tokens=len(self.tokenizer.encode(text)),
        )

    async def stream(self, prompt, max_tokens, temperature, top_k):
        from nanogpt import complete_stream

        async for piece in _aiter_blocking(
            complete_stream(self.model, self.tokenizer, prompt, max_tokens, temperature, top_k)
        ):
            yield piece


class QwenEngine(Engine):
    """Real inference over a HuggingFace Qwen3 checkpoint. Same boundary as
    NanoGPTEngine: transformers stays behind it, and blocking generation is pushed
    off the event loop so /health and /metrics answer during a completion."""

    def __init__(self, model, tokenizer, enable_thinking: bool = False):
        self.model, self.tokenizer = model, tokenizer
        self.enable_thinking = enable_thinking

    async def generate(self, prompt, max_tokens, temperature, top_k) -> Completion:
        from qwen import complete

        result = await asyncio.to_thread(
            complete,
            self.model,
            self.tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            self.enable_thinking,
        )
        # Unlike nano-GPT, the backend already counted tokens and knows why it
        # stopped -- this engine just renames the fields.
        return Completion(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

    async def stream(self, prompt, max_tokens, temperature, top_k):
        from qwen import complete_stream

        async for piece in _aiter_blocking(
            complete_stream(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens,
                temperature,
                top_k,
                self.enable_thinking,
            )
        ):
            yield piece


class BatchingEngine(Engine):
    """Qwen3 served through the continuous-batching scheduler in batching.py.

    Unlike the three engines above, this one does NOT generate anything itself.
    It submits a Sequence and then reads that sequence's queue -- all the actual
    work happens on the single background run_loop shared by every request. The
    engine is a per-request *view* onto a process-wide loop, which is why it is
    the one engine that owns its own concurrency accounting.
    """

    owns_concurrency = True

    def __init__(self, model, tokenizer, scheduler, enable_thinking: bool = False):
        from batching import resolve_eos_ids

        self.model, self.tokenizer = model, tokenizer
        self.scheduler = scheduler
        self.enable_thinking = enable_thinking
        # Resolved once: the stop-token set is a property of the model, not of a
        # request. Deriving it per call would re-read generation_config on every
        # HTTP hit and drag a model reference into a tokenisation helper.
        self.eos_token_ids = resolve_eos_ids(model, tokenizer)

    def stats(self) -> tuple[int, int]:
        return self.scheduler.num_running, self.scheduler.num_waiting

    def _submit(self, prompt, max_tokens, temperature, top_k):
        """Tokenize, wrap in a Sequence, hand to the scheduler. Returns the Sequence."""
        from batching import prompt_to_sequence

        seq = prompt_to_sequence(
            self.tokenizer, prompt, max_tokens, temperature, top_k,
            self.eos_token_ids, self.enable_thinking,
        )
        self.scheduler.add(seq)
        return seq

    def cancel(self, seq) -> None:
        """Abandon a sequence whose client went away. Safe on a finished one."""
        self.scheduler.abort(seq)

    # -----------------------------------------------------------------------
    # TODO 9 -- one-shot completion
    # -----------------------------------------------------------------------

    async def generate(self, prompt, max_tokens, temperature, top_k) -> Completion:
        """Submit, wait for the sentinel, return what the sequence accumulated.

        Shape:

            seq = self._submit(...)
            try:
                while await seq.queue.get() is not None:
                    pass                      # tokens land in seq.output_ids
            finally:
                self.cancel(seq)              # see below
            return Completion(...)

        Three things to get right:

        - The loop ends on the None sentinel, not on a count. Only the scheduler
          knows whether the sequence stopped on EOS or on max_new_tokens.
        - Decode the FULL output_ids at the end with skip_special_tokens=True.
          Do not accumulate the per-token pieces yourself; EOS is in output_ids
          (so the token count is honest) and must not reach the JSON body.
        - The `finally` is not optional. When an HTTP client disconnects, FastAPI
          cancels this coroutine and CancelledError is raised at the `await`. With
          no finally, the sequence keeps its batch slot and generates to
          max_new_tokens for a client that is already gone. cancel() is a no-op on
          a sequence that finished normally, so one unconditional call in finally
          covers both paths -- do not try to branch on why you left the loop.

        Counts come from the sequence, not from len(text) or max_tokens:
        prompt_tokens=len(seq.prompt_ids), completion_tokens=len(seq.output_ids),
        finish_reason=seq.finish_reason.
        """
        seq = self._submit(prompt, max_tokens, temperature, top_k)
        try:
            while await seq.queue.get() is not None:
                pass                     # token 自己会落进 seq.output_ids
        finally:
            self.cancel(seq)             # ← 不是可选项
        return Completion(
            text=self.tokenizer.decode(seq.output_ids, skip_special_tokens=True),
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids),
            finish_reason=seq.finish_reason,
        )

    # -----------------------------------------------------------------------
    # TODO 10 -- streaming
    # -----------------------------------------------------------------------

    async def stream(self, prompt, max_tokens, temperature, top_k):
        """Same submission, but yield text as it arrives.

        The trap here is detokenisation, and it is a real one for Qwen: a single
        CJK character or emoji is routinely 2-3 BPE tokens. Decoding each token
        on its own gives you U+FFFD replacement characters where the multi-byte
        characters should be -- fluent-looking English, mojibake everywhere else.

        Milestone 3 got this for free from TextIteratorStreamer. Driving the loop
        yourself means doing it yourself: decode the WHOLE output_ids each time,
        track how many characters you have already sent, and yield only the new
        suffix. An incomplete character contributes nothing to the decoded string,
        so it simply produces no yield that round and appears once complete.

            sent = 0
            while await seq.queue.get() is not None:
                text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)
                if len(text) > sent:
                    yield text[sent:]
                    sent = len(text)

        Wrap it in the same try/finally as TODO 9. Disconnects matter MORE here,
        not less: streaming clients are the ones that hang up early.
        """
        seq = self._submit(prompt, max_tokens, temperature, top_k)
        sent = 0

        try:
            while await seq.queue.get() is not None:
                text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)
                if len(text) > sent:
                    yield text[sent:]
                    sent = len(text)
        finally:
            self.cancel(seq)


# ---------------------------------------------------------------------------
# TODO 11 -- the metrics switchover
# ---------------------------------------------------------------------------


def engine_stats(engine: Engine, runtime: dict) -> tuple[int, int]:
    """(running, waiting) for /metrics, from whichever side is doing the counting.

    Return engine.stats() when the engine owns concurrency, otherwise read
    runtime["running"] / runtime["waiting"] -- the counters main.py's _slot()
    maintains for the engines that do not schedule.

    This function is four lines and it is the point of the entire milestone.
    Before it, `vllm:num_requests_waiting` reported "how many requests are stuck
    outside a semaphore whose size I invented". After it, the same metric, with
    the same name and the same Grafana panel, reports "how many requests could
    not start because the KV cache is full". The number finally measures
    something. That is what the README's autoscaling story was always assuming.

    It lives here rather than inline in main.py so it can be graded without
    standing up an ASGI app -- and so the branch has a name.
    """
    raise NotImplementedError("TODO 11")


def load_nanogpt(path):
    from nanogpt import load_model

    return load_model(path)


def load_qwen(model_id: str):
    from qwen import load_qwen as load

    return load(model_id)
