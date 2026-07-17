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
    assert "8 tokens" in choice["text"]


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


async def test_completion_missing_body_is_422(make_client):
    _, client = await make_client()
    resp = await client.post("/v1/completions")
    assert resp.status_code == 422


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

    reassembled = "".join(c["choices"][0]["text"] for c in chunks).strip()
    assert reassembled == "[mock output of 4 tokens]"


# --- 4. concurrency accounting -------------------------------------------


async def test_concurrency_ceiling_and_drain(make_client):
    _, client = await make_client(Settings(max_concurrency=2))
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
    _, client = await make_client(Settings(max_concurrency=2))
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
    _, client = await make_client(Settings(max_concurrency=2))
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
    _, client = await make_client(Settings(max_concurrency=1))

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
    _, client = await make_client(Settings(max_concurrency=1))
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
    assert body["choices"][0]["text"] == "[mock output of 0 tokens]"


async def test_instances_do_not_share_state(make_client):
    _, client_a = await make_client(Settings(model_name="a", max_concurrency=1))
    _, client_b = await make_client(Settings(model_name="b", max_concurrency=1))

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
