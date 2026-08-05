"""HTTP gateway. Routing, OpenAI schema translation, SSE framing, lifecycle.

Nothing here schedules, batches, counts, or knows what a token is. The test for
whether a line belongs in this file: would it still be needed if the frontend
were gRPC or an offline CLI? If yes, it belongs in the engine.

That rule is why the semaphore is gone. Milestones 1-3 kept concurrency
accounting here (`_slot`, `runtime["running"]`, `MAX_CONCURRENCY`) because each
request generated independently. With one scheduler owning admission, a second
gate here would mean two ceilings, two queues, and a metric reporting whichever
one happened to bind first.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from batching import Scheduler, run_loop
from engines import BatchingEngine
from runners import MockRunner

# Models under this prefix are served by the mock runner. A prefix rather than a
# single sentinel so several mock ModelServers can coexist under distinct names --
# "mock/fable-5", "mock/kimi-k3" -- which is what the Pending/Loading/Ready and
# queue-depth demos want. spec.model is required and non-empty on the CRD and the
# controller passes it through as MODEL_ID, so a mock CR cannot leave it blank;
# without this check it reaches load_qwen_runner and the mock image, which ships
# no torch, dies with an ImportError inside the un-awaited _load() task -- leaving
# /health on 503 forever with nothing in the logs to explain it.
MOCK_MODEL_PREFIX = "mock/"

# Reasons a sequence can end that are NOT an OpenAI finish_reason. The schema
# admits "stop", "length", "content_filter" and "tool_calls" only, so passing
# these through would put a value on the wire that a strict client rejects while
# parsing -- and would dress a dead run_loop up as a completed answer.
# "cancelled" is here for completeness; it means the client already left.
FATAL_FINISH_REASONS = {"error", "cancelled"}


@dataclass
class Settings:
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "unknown"))
    model_id: str = field(default_factory=lambda: os.environ.get("MODEL_ID", ""))
    load_time_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LOAD_TIME_SECONDS", "20"))
    )
    # The real KV-cache capacity limit, replacing MAX_CONCURRENCY. Unlike that
    # invented ceiling, this is what actually decides whether a request can start,
    # which is what makes vllm:num_requests_waiting a measurement.
    max_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("MAX_BATCH_SIZE", "8"))
    )
    max_batch_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_BATCH_TOKENS", "8192"))
    )
    enable_thinking: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_THINKING", "").lower() == "true"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    # Factory rather than a module-level app: each call gets its own Settings, its
    # own scheduler and its own loop task, so tests can spin up isolated instances
    # without env vars or module reloads.
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The mock serves immediately, on the same scheduler and the same loop the
        # real runner will use -- so behaviour during the Loading window is not a
        # special case, just a different runner.
        app.state.loop_task = asyncio.create_task(
            run_loop(app.state.engine.runner, app.state.scheduler)
        )

        async def _load():
            # Naming a fake model keeps "configuring a model IS the selection"
            # intact, rather than adding the engine switch that rule avoids.
            if settings.model_id and not settings.model_id.startswith(MOCK_MODEL_PREFIX):
                from runners import load_qwen_runner

                runner = await asyncio.to_thread(
                    load_qwen_runner, settings.model_id, settings.enable_thinking
                )
                # Swap runner and restart the loop against it. Nothing else in the
                # process changes: same scheduler, same engine, same metrics.
                app.state.loop_task.cancel()
                app.state.engine.runner = runner
                app.state.loop_task = asyncio.create_task(
                    run_loop(runner, app.state.scheduler)
                )
            # Kept even for the mock: the Pending->Loading->Ready demo needs a
            # visible window, and a real load takes a minute or two anyway.
            await asyncio.sleep(settings.load_time_seconds)
            app.state.runtime["loaded"] = True

        # Background task, not awaited: uvicorn must start listening now, so probes
        # get a 503 (up but not ready) rather than a connection refused.
        asyncio.create_task(_load())
        yield
        app.state.loop_task.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime = {"loaded": False, "started_at": time.time()}
    app.state.scheduler = Scheduler(settings.max_batch_size, settings.max_batch_tokens)
    app.state.engine = BatchingEngine(MockRunner(), app.state.scheduler)

    @app.get("/health")
    async def health():
        if not app.state.runtime["loaded"]:
            return Response(status_code=503)  # -> readiness fails -> Loading phase
        return {"status": "ok", "model": settings.model_name}

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 64))
        stream = bool(body.get("stream", False))
        temperature = float(body.get("temperature", 0.8))
        top_k = body.get("top_k")  # vLLM extension, not in the OpenAI schema

        cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if stream:
            return StreamingResponse(
                _stream(cmpl_id, created, prompt, max_tokens, temperature, top_k),
                media_type="text/event-stream",
            )

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
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    async def _stream(cmpl_id, created, prompt, max_tokens, temperature, top_k):
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

        async for piece in app.state.engine.stream(prompt, max_tokens, temperature, top_k):
            if piece.finish_reason in FATAL_FINISH_REASONS:
                # No [DONE], and the response body stops mid-stream. That IS the
                # error signal: the SSE framing has no error event, and the
                # status line went out with the first chunk, so a 500 is no
                # longer available. Reporting a normal finish_reason here would
                # hand the client a truncated answer it believes is complete.
                raise RuntimeError(f"generation failed: {piece.finish_reason}")
            yield chunk(piece.text, piece.finish_reason)
        yield "data: [DONE]\n\n"

    @app.get("/metrics")
    async def metrics():
        # Prometheus text format. Metric and label names copy vllm (model_name, not
        # model) so the monitoring side needs no change. One source now: the
        # scheduler that actually decides admission.
        running, waiting = app.state.engine.stats()
        text = (
            "# HELP vllm:num_requests_running Number of requests currently running on GPU.\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f'vllm:num_requests_running{{model_name="{settings.model_name}"}} {running}\n'
            "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            f'vllm:num_requests_waiting{{model_name="{settings.model_name}"}} {waiting}\n'
        )
        return Response(content=text, media_type="text/plain")

    return app


# Module-level instance for `uvicorn main:app` / the Dockerfile CMD.
app = create_app()
