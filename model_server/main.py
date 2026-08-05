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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

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

# Below this, temperature means "greedy" rather than "divide the logits by an
# almost-zero number". vLLM and SGLang both carry this constant (_SAMPLING_EPS,
# 1e-5 and 1e-6 respectively) for the same reason: temperature=1e-40 is not a
# malformed request, it is what a client sends when it wants determinism, and
# without a floor it produces inf logits, a NaN probability tensor, and a
# RuntimeError from multinomial deep inside the batch.
SAMPLING_EPS = 1e-5


class CompletionRequest(BaseModel):
    """Admission validation: everything the engine may not be asked to survive.

    This is the layer both vLLM and SGLang put in FRONT of the engine --
    SamplingParams._verify_args / SamplingParams.verify -- and the reason a bad
    request costs one 400 there instead of the process. Before it existed here,
    `top_k` went from JSON straight into torch.topk, so {"top_k": "5"} raised a
    TypeError on a WORKER THREAD, inside a forward pass shared with seven other
    requests, where the only available response was to fail the whole batch.

    Unknown fields are ignored rather than rejected: real clients send `n`,
    `echo`, `stop`, `logit_bias` and more, and 400-ing an OpenAI SDK for sending
    what the OpenAI SDK sends would be a worse contract than ignoring it.
    """

    prompt: str = ""
    max_tokens: int = Field(default=64, ge=0)
    # Upper bound copied from vLLM: beyond ~2 the distribution is noise, and a
    # finite cap is also what stops `inf` and `nan` from arriving as floats.
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    # -1 (SGLang) and 0 (vLLM) both spell "disabled"; anything below -1 is a
    # typo, not a convention.
    top_k: int | None = Field(default=None, ge=-1)
    stream: bool = False

    @model_validator(mode="after")
    def _normalize(self):
        """Fold the disabled/greedy spellings into the ones _sample understands.

        Normalising rather than rejecting, because both engines normalise here:
        a client asking for temperature=1e-40 wants greedy decoding and should
        get it, not a 400 explaining that its zero is insufficiently zero.
        """
        if self.temperature < SAMPLING_EPS:
            self.temperature = 0.0  # -> argmax in ModelRunner._sample
        if self.top_k is not None and self.top_k <= 0:
            self.top_k = None
        return self


def error_response(status: int, message: str, err_type: str, param: str | None = None):
    """The OpenAI error envelope: {"error": {message, type, param, code}}.

    Not a completion with a bad finish_reason, which is what this server used to
    return -- HTTP 200 carrying finish_reason="error" told an SDK the request had
    succeeded, while the value itself was outside the schema's enum.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type,
                           "param": param, "code": None}},
    )


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

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request, exc: RequestValidationError):
        """FastAPI's 422 is not the OpenAI contract; a bad request is a 400.

        Only the first error is reported, with `param` naming the field, which
        is the shape an OpenAI client already knows how to display.
        """
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first["loc"][1:]) or None
        return error_response(400, first["msg"], "invalid_request_error", param)

    def loop_is_dead() -> bool:
        """True once nothing is draining the scheduler.

        run_loop's contract is to die loudly on an unrecoverable error, and its
        docstring says "let Kubernetes restart the pod" -- but nothing used to
        SAY so. /health stayed 200, readiness kept passing, the Service kept
        routing, and every arriving request blocked on a queue no one would ever
        drain: a pod that accepts work and answers none of it, indefinitely.
        """
        task = getattr(app.state, "loop_task", None)
        return task is None or task.done()

    @app.get("/health")
    async def health():
        if not app.state.runtime["loaded"]:
            return Response(status_code=503)  # -> readiness fails -> Loading phase
        if loop_is_dead():
            return Response(status_code=503)  # -> out of the Service's endpoints
        return {"status": "ok", "model": settings.model_name}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        # Fail fast rather than accept work nobody will do. vLLM does the same
        # once its EngineCore dies: in-flight requests error out and new ones are
        # refused, instead of every client discovering the outage by timeout.
        if loop_is_dead():
            return error_response(
                503, "the inference loop is not running", "service_unavailable"
            )

        prompt, max_tokens = body.prompt, body.max_tokens
        temperature, top_k = body.temperature, body.top_k

        cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.stream:
            return StreamingResponse(
                _stream(cmpl_id, created, prompt, max_tokens, temperature, top_k),
                media_type="text/event-stream",
            )

        result = await app.state.engine.generate(prompt, max_tokens, temperature, top_k)
        if result.finish_reason in FATAL_FINISH_REASONS:
            # Nothing has been written yet on this path, so unlike the streaming
            # case a real status code is still available. Use it.
            return error_response(
                500, f"generation failed: {result.finish_reason}", "internal_server_error"
            )

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
