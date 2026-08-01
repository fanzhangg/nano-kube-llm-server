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
    implementation, not touching a single handler. The server owns concurrency
    accounting (the _slot semaphore); the engine owns only "how to generate"."""

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


def load_nanogpt(path):
    from nanogpt import load_model

    return load_model(path)


def load_qwen(model_id: str):
    from qwen import load_qwen as load

    return load(model_id)
