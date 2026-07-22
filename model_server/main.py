import asyncio
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse


@dataclass
class Settings:
    checkpoint_path: str = field(
        default_factory=lambda: os.environ.get("CHECKPOINT_PATH", "")
    )
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "unknown"))
    load_time_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LOAD_TIME_SECONDS", "20"))
    )
    # Fakes vLLM's scheduler capacity: in the real engine this ceiling comes from how many
    # concurrent sequences the KV cache can hold. A ceiling is what creates a queue, and a
    # queue is what makes num_requests_waiting > 0 -- the exact signal the autoscaler watches.
    max_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONCURRENCY", "4"))
    )

@dataclass
class Completion:
    """What an Engine returns: the generated text plus honest token counts. The
    engine owns tokenization, so it's the only thing that can count truthfully --
    the HTTP layer just copies these into the usage block."""

    text: str
    prompt_tokens: int
    completion_tokens: int


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

        gen = complete_stream(
            self.model, self.tokenizer, prompt, max_tokens, temperature, top_k
        )
        # complete_stream is a blocking generator -- each next() is one real forward
        # pass. Pull items one at a time via to_thread so the event loop stays free
        # between decode steps. The sentinel marks exhaustion (avoids raising
        # StopIteration across a coroutine boundary, which asyncio mangles).
        done = object()
        while True:
            piece = await asyncio.to_thread(next, gen, done)
            if not isinstance(piece, str):  # the sentinel -> generator is exhausted
                break
            yield piece


def _load_real_model(path):
    from nanogpt import load_model

    return load_model(path)


def create_app(settings: Settings | None = None) -> FastAPI:
    # Factory instead of a module-level app: each call gets its own Settings, its own
    # runtime state, and its own semaphore, so tests can spin up isolated instances
    # (short load time, small concurrency ceiling, distinct model name) without env
    # vars or module reloads, and without one test's in-flight requests leaking into
    # another's counters.
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _load():
            if settings.checkpoint_path:
                model, tokenizer = await asyncio.to_thread(
                    _load_real_model, settings.checkpoint_path
                )
                app.state.engine = NanoGPTEngine(model, tokenizer)
            # Keep the sleep even for a real checkpoint: loading a 3MB nano-GPT is
            # near-instant, but the Pending->Loading->Ready demo needs a visible
            # window. Semantics shift from "fake loading" to "artificial floor" --
            # a real vLLM load of a 0.5B model takes a minute or two anyway.
            await asyncio.sleep(settings.load_time_seconds)
            app.state.runtime["loaded"] = True

        # Kick loading into a background task and return immediately. If we awaited it
        # here, uvicorn would not start listening until loading finished, so probes
        # would get a connection refused (process not up) instead of a 503 (up but not
        # ready) -- very different semantics.
        asyncio.create_task(_load())
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    # In-memory per-app state and per-app semaphore: set at creation time (not inside
    # lifespan) so they exist for any ASGI harness, including ones that never run
    # startup events. Only the loading simulation itself needs the lifespan hook.
    app.state.runtime = {"loaded": False, "started_at": time.time(), "running": 0, "waiting": 0}
    app.state.slots = asyncio.Semaphore(settings.max_concurrency)
    # Default to the mock engine so there's always an engine to call. If a checkpoint
    # is configured, _load swaps in a NanoGPTEngine once weights are ready -- handlers
    # never need an "is the model loaded yet?" branch, they just call app.state.engine.
    app.state.engine = MockEngine(settings.model_name)

    @app.get("/health")
    async def health():
        # Model not loaded yet -> 503 -> readiness fails -> Loading phase
        if not app.state.runtime["loaded"]:
            return Response(status_code=503)
        return {"status": "ok", "model": settings.model_name}

    @asynccontextmanager
    async def _slot():
        # A request enters the queue (waiting), then counts as running once it grabs a
        # slot. The handoff must happen at the moment the semaphore is acquired so that
        # waiting + running always equals the number of in-flight requests. The work is
        # done while holding the slot, which is what makes the concurrency ceiling
        # actually bite.
        runtime = app.state.runtime
        runtime["waiting"] += 1
        acquired = False
        try:
            async with app.state.slots:
                runtime["waiting"] -= 1
                acquired = True
                runtime["running"] += 1
                try:
                    yield
                finally:
                    runtime["running"] -= 1
        finally:
            # Only reached when the request was cancelled while still queued (client
            # disconnect). Otherwise the handoff above already decremented waiting, and
            # decrementing twice would drive the counter negative.
            if not acquired:
                runtime["waiting"] -= 1

    @app.post("/v1/completions")
    async def completions(body: dict):
        # OpenAI legacy text-completion contract: read prompt / max_tokens / stream from
        # the request, and fill in the response fields (id/object/created/
        # choices[].finish_reason/usage) per the official schema so the official openai
        # SDK can parse the response straight into a Completion object.
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 64))
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.8))
        top_k = body.get("top_k")  # vLLM extension, not in the OpenAI schema; optional

        cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            return StreamingResponse(
                _stream(cmpl_id, created, prompt, max_tokens, temperature, top_k),
                media_type="text/event-stream",
            )

        # The server owns concurrency accounting (the slot); the engine owns
        # generation. Handlers no longer know whether it's mock or real.
        async with _slot():
            result = await app.state.engine.generate(prompt, max_tokens, temperature, top_k)

        return {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": settings.model_name,
            "choices": [
                {
                    "text": result.text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    async def _stream(cmpl_id, created, prompt, max_tokens, temperature, top_k):
        # SSE streaming: each chunk is `data: {json}\n\n`, terminated by
        # `data: [DONE]\n\n`, mirroring OpenAI so the official SDK's stream=True can
        # iterate it directly. The slot is held for the whole generation so the
        # counters stay accurate.
        def chunk(text: str, finish_reason):
            payload = {
                "id": cmpl_id,
                "object": "text_completion",
                "created": created,
                "model": settings.model_name,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            return f"data: {json.dumps(payload)}\n\n"

        async with _slot():
            async for piece in app.state.engine.stream(prompt, max_tokens, temperature, top_k):
                yield chunk(piece, None)
            yield chunk("", "length")
        yield "data: [DONE]\n\n"

    @app.get("/metrics")
    async def metrics():
        # Prometheus text format: one `name{label="value"} number` per line, newline-
        # terminated, text/plain. Metric and label names copy vllm (it's model_name, not
        # model) so the monitoring side needs no change when swapping in the real engine.
        runtime = app.state.runtime
        text = (
            "# HELP vllm:num_requests_running Number of requests currently running on GPU.\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f'vllm:num_requests_running{{model_name="{settings.model_name}"}} {runtime["running"]}\n'
            "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            f'vllm:num_requests_waiting{{model_name="{settings.model_name}"}} {runtime["waiting"]}\n'
        )
        return Response(content=text, media_type="text/plain")

    return app


# Module-level instance for `uvicorn main:app` / the Dockerfile CMD. Reads config from
# env vars exactly as before -- the external interface is unchanged.
app = create_app()
