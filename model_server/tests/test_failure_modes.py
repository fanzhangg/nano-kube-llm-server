"""What a client sees when the engine dies mid-flight, over a real socket.

test_main.py grades the same handlers through httpx.ASGITransport, which calls
the app in-process: no sockets, no uvicorn, no chunked transfer encoding. That
is enough to prove a handler returns 503, and not enough to prove anything about
a response whose headers have already been sent -- which is exactly the case
here. Once the first SSE chunk is on the wire the status code is spent, so
"how does this request fail" becomes a question about framing, and framing only
exists over a real server.

MockRunner, deliberately: none of this is model-specific, and the mock starts
instantly, so these run in the default suite rather than behind an opt-in mark.
The Qwen-specific half of the failure story (admission validation with real
weights behind it) lives in test_e2e_qwen.py.
"""

import json
import threading
import time

import httpx
import pytest

from conftest import serve
from main import Settings, create_app


@pytest.fixture
def live_mock():
    """A real uvicorn on a real port, serving the mock runner."""
    app = create_app(Settings(model_name="mock/failure", load_time_seconds=0,
                              max_batch_size=4))
    with serve(app, ready_timeout=30) as base_url:
        yield app, base_url


def kill_the_loop(app) -> None:
    """Cancel run_loop from OUTSIDE the server's event loop, as a crash would.

    call_soon_threadsafe rather than task.cancel(): the task belongs to the
    server thread's loop, and touching a Future from another thread is the same
    unsupported crossing Sequence.emit exists to avoid. run_loop turns the
    cancellation into fail_all + re-raise -- the identical path a CUDA OOM takes.
    """
    loop = app.state.loop_task.get_loop()
    loop.call_soon_threadsafe(app.state.loop_task.cancel)


def test_a_dying_loop_breaks_an_open_stream_instead_of_hanging(live_mock):
    """The one failure the in-process tests cannot reach.

    Headers and the first chunks are already sent, so there is no status code
    left to fail with. The stream must terminate WITHOUT its [DONE] sentinel:
    that truncation is the only signal available, and it is what tells a client
    the answer it holds is partial. The alternative -- a connection that stays
    open forever behind a dead loop -- is indistinguishable from a slow model,
    and clients wait it out.
    """
    app, base_url = live_mock
    chunks, done_seen = [], False

    with httpx.Client(base_url=base_url, timeout=30) as http:
        with http.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 500, "stream": True},
        ) as response:
            assert response.status_code == 200
            try:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        done_seen = True
                        break
                    chunks.append(line)
                    if len(chunks) == 3:
                        threading.Thread(target=kill_the_loop, args=(app,)).start()
            except httpx.HTTPError:
                pass  # a torn-off body is a legitimate way to observe this

    assert len(chunks) >= 3, "the stream never started; nothing was under test"
    assert not done_seen, "a stream cut short by a dead engine must not say [DONE]"


def test_a_dying_loop_is_visible_to_the_kubelet(live_mock):
    """/health over a socket is what the probes actually poll.

    The startup probe has long since succeeded by the time this happens, so
    `loaded` is true and this 503 can only mean the loop is gone -- which is
    precisely the distinction the liveness/startup split in the controller
    depends on. Restart is the remedy; being asked is the prerequisite.
    """
    app, base_url = live_mock

    with httpx.Client(base_url=base_url, timeout=30) as http:
        assert http.get("/health").status_code == 200

        kill_the_loop(app)
        for _ in range(100):
            if http.get("/health").status_code == 503:
                break
            time.sleep(0.05)

        assert http.get("/health").status_code == 503


def test_a_dying_loop_refuses_new_work_over_the_wire(live_mock):
    """503 with the OpenAI error envelope, not a request that never returns.

    Timed, not just status-checked: the bug this replaced was a request that
    blocked forever on a queue nobody drained, and "returns quickly" is half of
    what is being asserted.
    """
    app, base_url = live_mock

    with httpx.Client(base_url=base_url, timeout=10) as http:
        kill_the_loop(app)
        for _ in range(100):
            if http.get("/health").status_code == 503:
                break
            time.sleep(0.05)

        started = time.perf_counter()
        response = http.post("/v1/completions", json={"prompt": "hi", "max_tokens": 8})
        elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable"
    assert elapsed < 2, f"took {elapsed:.1f}s to refuse -- clients see that as a hang"


def test_a_rejected_request_never_reaches_the_scheduler(live_mock):
    """A 400 must cost nothing: no slot, no queue entry, no metric movement.

    Admission validation is only containment if it happens BEFORE the sequence
    is submitted. Validating after would still return 400 while leaving a
    request the scheduler intends to run and nobody intends to read.
    """
    app, base_url = live_mock

    with httpx.Client(base_url=base_url, timeout=10) as http:
        rejected = http.post("/v1/completions", json={"prompt": "hi", "top_k": 2.5})
        metrics = http.get("/metrics").text

    assert rejected.status_code == 400
    assert json.loads(rejected.text)["error"]["param"] == "top_k"
    assert 'vllm:num_requests_running{model_name="mock/failure"} 0' in metrics
    assert 'vllm:num_requests_waiting{model_name="mock/failure"} 0' in metrics
