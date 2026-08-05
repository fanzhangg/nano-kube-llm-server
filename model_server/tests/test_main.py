"""Tests for the mock inference server.

These test the *contract* the mock exposes to its consumers: the K8s
controller (health/readiness), the autoscaler (metrics), and the OpenAI
API surface (/v1/completions). They deliberately don't test whether the
generated text is realistic -- that's meaningless for a mock, and stops
mattering entirely once this is swapped for a real inference engine. What
does carry over is the contract: response shape, metric semantics, and
concurrency accounting.
"""

import asyncio
import json
from contextlib import AsyncExitStack

import httpx
import pytest

from main import Settings, create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def make_client():
    """Factory for isolated (app, client) pairs.

    Each call gets its own Settings, its own runtime state, and its own
    concurrency semaphore (see Settings/create_app in main.py), so tests
    can use whatever load_time_seconds/max_concurrency/model_name they
    need without touching env vars or leaking state into other tests.
    """
    async with AsyncExitStack() as stack:

        async def _make(settings: Settings | None = None):
            app = create_app(settings)
            await stack.enter_async_context(app.router.lifespan_context(app))
            client = await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                )
            )
            return app, client

        yield _make


def metric_value(text: str, name: str) -> int:
    for line in text.splitlines():
        if line.startswith(name + "{"):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} not found in:\n{text}")


# --- 1. lifecycle / /health --------------------------------------------


async def test_health_503_before_loaded(make_client):
    _, client = await make_client(Settings(load_time_seconds=999))
    resp = await client.get("/health")
    assert resp.status_code == 503


async def test_health_200_after_loaded(make_client):
    _, client = await make_client(Settings(load_time_seconds=0.05, model_name="qwen-x"))
    await asyncio.sleep(0.2)
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "qwen-x"}


async def test_mock_prefix_keeps_the_mock_engine(make_client):
    """A "mock/" model id must not attempt a real load.

    spec.model is required and non-empty on the CRD, and the controller passes
    it through as MODEL_ID, so a mock ModelServer cannot leave it blank. Without
    the prefix check this reaches load_qwen and the mock image -- which ships no
    torch -- fails with ImportError inside the un-awaited _load() task, leaving
    /health on 503 forever with nothing in the logs to explain it.
    """
    app, client = await make_client(
        Settings(model_id="mock/qwen-small", model_name="mock/qwen-small",
                 load_time_seconds=0.05)
    )
    await asyncio.sleep(0.2)

    assert type(app.state.engine.runner).__name__ == "MockRunner"
    assert (await client.get("/health")).status_code == 200

    resp = await client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4})
    assert resp.status_code == 200
    assert resp.json()["model"] == "mock/qwen-small"


# --- 2. non-streaming /v1/completions -----------------------------------


async def test_completion_response_shape(make_client):
    _, client = await make_client(Settings(model_name="qwen-x"))
    resp = await client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert isinstance(body["created"], int) and body["created"] > 0
    assert body["model"] == "qwen-x"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "length"
    assert choice["text"] == "mock " * 8


async def test_completion_usage_accounting(make_client):
    _, client = await make_client()
    resp = await client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 10})
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == len("hello")
    assert usage["completion_tokens"] == 10
    assert usage["total_tokens"] == len("hello") + 10


async def test_completion_defaults_when_fields_omitted(make_client):
    _, client = await make_client()
    resp = await client.post("/v1/completions", json={})
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 0  # prompt defaults to ""
    assert usage["completion_tokens"] == 64  # max_tokens defaults to 64


async def test_completion_missing_body_is_400(make_client):
    """400, not FastAPI's default 422: the OpenAI contract for a bad request."""
    _, client = await make_client()
    resp = await client.post("/v1/completions")
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# --- 2b. admission validation -------------------------------------------
#
# Everything below used to reach the sampler, on a worker thread, inside a
# forward pass shared with every other in-flight request -- where the only
# response available was to fail the whole batch and kill the loop. vLLM and
# SGLang both validate in front of the engine for exactly this reason.


@pytest.mark.parametrize(
    "body,param",
    [
        ({"prompt": "hi", "top_k": "not-an-int"}, "top_k"),
        ({"prompt": "hi", "top_k": 2.5}, "top_k"),          # -> torch.topk TypeError
        ({"prompt": "hi", "top_k": -7}, "top_k"),           # -1/0 mean disabled; -7 is a typo
        ({"prompt": "hi", "max_tokens": -1}, "max_tokens"),
        ({"prompt": "hi", "temperature": -0.5}, "temperature"),
        ({"prompt": "hi", "temperature": 99}, "temperature"),
        ({"prompt": ["a", "b"]}, "prompt"),
    ],
)
async def test_invalid_sampling_params_are_rejected_at_admission(make_client, body, param):
    _, client = await make_client()

    resp = await client.post("/v1/completions", json=body)

    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param


async def test_a_rejected_request_leaves_the_server_serving(make_client):
    """The whole point. A 400 must cost one request, not the process.

    Before admission validation, {"top_k": "5"} raised inside the shared forward
    pass, fail_all took down every in-flight request, run_loop died, /health
    stayed 200, and every later request hung forever on a queue nobody drained.
    """
    # load_time_seconds=0 so /health reports the LOOP's health here rather than
    # the loading window it also multiplexes.
    _, client = await make_client(Settings(load_time_seconds=0))
    await asyncio.sleep(0.05)

    rejected = await client.post("/v1/completions", json={"prompt": "hi", "top_k": "x"})
    assert rejected.status_code == 400

    after = await client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4})
    assert after.status_code == 200
    assert (await client.get("/health")).status_code == 200


async def test_a_near_zero_temperature_means_greedy_not_a_crash(make_client):
    """1e-40 is not malformed -- it is a client asking for determinism.

    Rejecting it would be defensible; crashing is not, and dividing logits by
    1e-40 gives inf, then a NaN probability tensor, then a RuntimeError from
    multinomial in the middle of someone else's batch. Both vLLM and SGLang
    fold it to greedy instead (_SAMPLING_EPS), which is what this asserts.
    """
    _, client = await make_client()

    resp = await client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 4, "temperature": 1e-40}
    )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "length"


async def test_unknown_fields_are_ignored_not_rejected(make_client):
    """Real OpenAI clients send fields this server does not implement."""
    _, client = await make_client()

    resp = await client.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": 2, "model": "qwen", "n": 1, "echo": False},
    )

    assert resp.status_code == 200


# --- 2c. failing fast when the blast radius really is everyone ----------


async def test_a_dead_loop_fails_readiness(make_client):
    """run_loop's docstring says "let Kubernetes restart the pod" -- this is what says so.

    While /health stayed 200 with a dead loop, the Service kept routing to a pod
    that could not answer, and requests hung instead of failing. Readiness is
    the signal that takes it out of the endpoints.
    """
    app, client = await make_client(Settings(load_time_seconds=0))
    await asyncio.sleep(0.05)
    assert (await client.get("/health")).status_code == 200

    app.state.loop_task.cancel()
    await asyncio.sleep(0.05)

    assert (await client.get("/health")).status_code == 503


async def test_a_dead_loop_refuses_new_work_instead_of_hanging(make_client):
    """503 now beats a request that blocks until the client's timeout.

    Accepting work into a scheduler nobody drains is the failure that hid the
    original bug: no error, no 500, no log line -- just requests that never
    came back.
    """
    app, client = await make_client(Settings(load_time_seconds=0))
    await asyncio.sleep(0.05)
    app.state.loop_task.cancel()
    await asyncio.sleep(0.05)

    resp = await asyncio.wait_for(
        client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4}), timeout=2
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "service_unavailable"


async def test_a_failed_generation_is_a_500_not_a_200(make_client):
    """finish_reason="error" is not an OpenAI value and not a success.

    A crashed loop used to answer 200 with an empty completion and
    finish_reason="error": outside the schema's enum, wrapped in a success, so
    an SDK reported that the request had worked.
    """
    app, client = await make_client(Settings(load_time_seconds=0))
    await asyncio.sleep(0.05)

    task = asyncio.create_task(
        client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 200})
    )
    await asyncio.sleep(0.05)
    app.state.scheduler.fail_all(RuntimeError("CUDA out of memory"))
    resp = await asyncio.wait_for(task, timeout=2)

    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "internal_server_error"


# --- 3. streaming /v1/completions ---------------------------------------


async def test_streaming_shape_and_termination(make_client):
    _, client = await make_client()
    async with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": 4, "stream": True},
    ) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in resp.aiter_lines() if line]

    data_lines = [line[len("data: ") :] for line in lines if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"

    chunks = [json.loads(d) for d in data_lines[:-1]]
    assert len({c["id"] for c in chunks}) == 1  # every chunk shares one completion id
    assert all(c["object"] == "text_completion" for c in chunks)
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [None] * (
        len(chunks) - 1
    ) + ["length"]

    # No .strip(): the stream must reassemble byte-for-byte into what the
    # non-streaming path returns, trailing space included.
    reassembled = "".join(c["choices"][0]["text"] for c in chunks)
    assert reassembled == "mock " * 4  # one piece per decode tick


# --- 4. concurrency accounting -------------------------------------------


async def test_concurrency_ceiling_and_drain(make_client):
    _, client = await make_client(Settings(max_batch_size=2))
    tasks = [
        asyncio.create_task(
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 100})
        )
        for _ in range(5)
    ]
    await asyncio.sleep(0.05)
    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 2
    assert metric_value(metrics, "vllm:num_requests_waiting") == 3

    await asyncio.gather(*tasks)
    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 0
    assert metric_value(metrics, "vllm:num_requests_waiting") == 0


async def test_cancel_while_queued_does_not_leak_waiting(make_client):
    _, client = await make_client(Settings(max_batch_size=2))
    tasks = [
        asyncio.create_task(
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 200})
        )
        for _ in range(5)
    ]
    await asyncio.sleep(0.05)  # 2 running, 3 queued
    tasks[4].cancel()
    try:
        await tasks[4]
    except (asyncio.CancelledError, httpx.HTTPError):
        pass

    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_waiting") == 2

    for t in tasks[:4]:
        t.cancel()
    await asyncio.gather(*tasks[:4], return_exceptions=True)
    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 0
    assert metric_value(metrics, "vllm:num_requests_waiting") == 0


async def test_cancel_while_running_does_not_leak_running(make_client):
    _, client = await make_client(Settings(max_batch_size=2))
    tasks = [
        asyncio.create_task(
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 200})
        )
        for _ in range(2)
    ]
    await asyncio.sleep(0.05)  # both running
    tasks[0].cancel()
    try:
        await tasks[0]
    except (asyncio.CancelledError, httpx.HTTPError):
        pass

    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 1

    tasks[1].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 0


async def test_streaming_request_counts_toward_running(make_client):
    _, client = await make_client(Settings(max_batch_size=1))

    async def consume_stream():
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "x", "max_tokens": 200, "stream": True},
        ) as resp:
            async for _ in resp.aiter_lines():
                pass

    stream_task = asyncio.create_task(consume_stream())
    await asyncio.sleep(0.05)

    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_running") == 1

    # A second request must queue behind the streaming one, not bypass it.
    second = asyncio.create_task(
        client.post("/v1/completions", json={"prompt": "y", "max_tokens": 4})
    )
    await asyncio.sleep(0.05)
    metrics = (await client.get("/metrics")).text
    assert metric_value(metrics, "vllm:num_requests_waiting") == 1

    await asyncio.gather(stream_task, second)


# --- 5. /metrics format ---------------------------------------------------


async def test_metrics_format_and_label(make_client):
    _, client = await make_client(Settings(model_name="qwen-x"))
    resp = await client.get("/metrics")
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert "# HELP vllm:num_requests_running" in text
    assert "# TYPE vllm:num_requests_running gauge" in text
    assert "# TYPE vllm:num_requests_waiting gauge" in text
    assert 'model_name="qwen-x"' in text
    # regression guard: the label used to be the bare `model`, not `model_name`
    assert "{model=" not in text


async def test_metrics_not_blocked_by_in_flight_completion(make_client):
    _, client = await make_client(Settings(max_batch_size=1))
    slow = asyncio.create_task(
        client.post("/v1/completions", json={"prompt": "x", "max_tokens": 500})
    )
    await asyncio.sleep(0.05)
    resp = await asyncio.wait_for(client.get("/metrics"), timeout=1)
    assert resp.status_code == 200
    slow.cancel()
    await asyncio.gather(slow, return_exceptions=True)


# --- 6. edge cases and instance isolation ----------------------------------


async def test_zero_max_tokens(make_client):
    _, client = await make_client()
    resp = await client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"]["completion_tokens"] == 0
    assert body["choices"][0]["text"] == ""


async def test_instances_do_not_share_state(make_client):
    _, client_a = await make_client(Settings(model_name="a", max_batch_size=1))
    _, client_b = await make_client(Settings(model_name="b", max_batch_size=1))

    busy_a = asyncio.create_task(
        client_a.post("/v1/completions", json={"prompt": "x", "max_tokens": 200})
    )
    await asyncio.sleep(0.05)

    metrics_a = (await client_a.get("/metrics")).text
    metrics_b = (await client_b.get("/metrics")).text
    assert metric_value(metrics_a, "vllm:num_requests_running") == 1
    assert metric_value(metrics_b, "vllm:num_requests_running") == 0

    busy_a.cancel()
    await asyncio.gather(busy_a, return_exceptions=True)
