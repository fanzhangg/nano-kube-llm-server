"""Shared fixtures, plus the machinery for running the app under a REAL server.

`httpx.ASGITransport` (test_main.py) calls the app in-process: no sockets, no
uvicorn, no SSE framing over the wire. That is the right tool for grading
handler logic, and the wrong one for grading a service -- it cannot show that
the stream flushes chunk by chunk rather than all at once at the end, and it
cannot be pointed at a pod. `serve()` below runs uvicorn on a real port, which
is what test_e2e_qwen.py and test_bench_serving.py measure, and it is also why
both files can be redirected at a deployed ModelServer with MODEL_SERVER_URL
and keep every assertion.
"""

import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import uvicorn


@pytest.fixture
def anyio_backend():
    return "asyncio"


def free_port() -> int:
    """Bind :0, read back what the kernel picked, release it.

    Racy in principle, fine in practice, and the only way to get a port without
    a fixed number that collides the moment two test files run at once.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_ready(base_url: str, timeout: float, is_alive=lambda: True) -> None:
    """Block until /health returns 200 -- i.e. the model is LOADED, not merely up.

    503 is the honest answer during loading (see main.create_app), so polling for
    "connection accepted" would hand back a server that 503s every request. The
    `is_alive` hook exists because a load failure inside the un-awaited _load()
    task leaves /health on 503 forever: without it, a broken model id turns into
    a ten-minute wait instead of an immediate error.
    """
    deadline = time.monotonic() + timeout
    last = "no attempt made"
    while time.monotonic() < deadline:
        if not is_alive():
            raise RuntimeError(f"{base_url}: server stopped before becoming ready")
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(0.25)
    raise RuntimeError(f"{base_url}: not ready after {timeout:.0f}s (last: {last})")


@contextmanager
def serve(app, ready_timeout: float = 60.0):
    """Run `app` under uvicorn on a free port; yield its base URL.

    Daemon thread rather than a subprocess so the test process can still reach
    app.state (the scheduler, the runner) when it wants to, and so a crashed
    test cannot leave a server behind.
    """
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_until_ready(base_url, ready_timeout, is_alive=thread.is_alive)
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def metric_value(text: str, name: str) -> int:
    """Pull one gauge out of the Prometheus exposition text."""
    for line in text.splitlines():
        if line.startswith(name + "{"):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} not found in:\n{text}")
