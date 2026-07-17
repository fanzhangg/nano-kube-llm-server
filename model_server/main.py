import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse


@dataclass
class Settings:
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
            await asyncio.sleep(settings.load_time_seconds)  # simulate model loading
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

    def _mock_text(max_tokens: int) -> str:
        return f"[mock output of {max_tokens} tokens]"

    @app.post("/v1/completions")
    async def completions(body: dict):
        # OpenAI legacy text-completion contract: read prompt / max_tokens / stream from
        # the request, and fill in the response fields (id/object/created/
        # choices[].finish_reason/usage) per the official schema so the official openai
        # SDK can parse the response straight into a Completion object.
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 64))
        stream = bool(body.get("stream", False))
        prompt_len = len(prompt)
        latency = prompt_len * 0.001 + max_tokens * 0.02  # simulate prefill + decode time

        cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            return StreamingResponse(
                _stream(cmpl_id, created, latency, max_tokens),
                media_type="text/event-stream",
            )

        async with _slot():
            await asyncio.sleep(latency)

        return {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": settings.model_name,
            "choices": [
                {
                    "text": _mock_text(max_tokens),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": max_tokens,
                "total_tokens": prompt_len + max_tokens,
            },
        }

    async def _stream(cmpl_id: str, created: int, latency: float, max_tokens: int):
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
            pieces = _mock_text(max_tokens).split(" ")
            per_piece = latency / max(len(pieces), 1)
            for piece in pieces:
                await asyncio.sleep(per_piece)
                yield chunk(piece + " ", None)
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
