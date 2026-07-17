"""Proves /v1/completions is actually OpenAI-compatible, not just JSON-shaped
like it. Runs a real uvicorn server and drives it with the official openai
SDK -- if a required response field is missing or malformed, the SDK's own
response parsing raises before these assertions ever run.
"""

import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from openai import OpenAI

from main import create_app, Settings


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    port = _free_port()
    app = create_app(Settings(model_name="qwen-live", load_time_seconds=0, max_concurrency=4))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        raise RuntimeError("live mock server never became ready")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def test_non_streaming_parses_as_completion(live_server):
    client = OpenAI(base_url=f"{live_server}/v1", api_key="not-needed")
    c = client.completions.create(model="qwen-live", prompt="hello world", max_tokens=8)
    assert c.id.startswith("cmpl-")
    assert c.object == "text_completion"
    assert c.model == "qwen-live"
    assert c.choices[0].finish_reason == "length"
    assert c.usage.completion_tokens == 8
    assert c.usage.prompt_tokens == len("hello world")


def test_streaming_iterates_to_completion(live_server):
    client = OpenAI(base_url=f"{live_server}/v1", api_key="not-needed")
    stream = client.completions.create(model="qwen-live", prompt="hi", max_tokens=8, stream=True)
    texts, last_finish = [], None
    for event in stream:
        assert event.object == "text_completion"
        texts.append(event.choices[0].text)
        last_finish = event.choices[0].finish_reason
    assert last_finish == "length"
    assert "mock output of 8 tokens" in "".join(texts)
