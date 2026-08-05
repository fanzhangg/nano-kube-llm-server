"""Serving benchmark: what a CLIENT experiences, under concurrency.

bench.py measures the MODEL -- one process, generate(), no HTTP, no scheduler.
It answers "how fast can these weights decode". This file measures the SERVICE:
requests arriving over a socket, admitted by the Scheduler, sharing forward
passes, streaming back. It answers the question an operator actually has --
"what do my users see at C concurrent requests, and where does it fall over" --
and those are different numbers. A service can hit bench.py's decode rate at
C=1 and still deliver 4-second TTFTs at C=16 because everything is queueing.

The metrics are vLLM's and SGLang's, so the numbers are comparable to published
ones:

    TTFT    time to first token -- prefill + queueing delay. What feels like lag.
    ITL     inter-token latency, first token excluded. What reading speed feels like.
    TPOT    mean ITL per request. 1/TPOT is per-user tok/s.
    E2E     full request latency. p99 is the one that gets you paged.
    output tok/s   tokens produced by the whole service per second. Capacity.

Three of the assertions here are hardware-independent -- they compare the
service against ITSELF at another concurrency level, which is exactly what
continuous batching claims and what a regression would break:

    batching lifts throughput          C=max_batch is >=2x C=1
    latency stays sublinear            per-request latency does not scale with C
    the tail stays bounded             p99/p50 under uniform load

The absolute budgets (TTFT/throughput) are opt-in via env, because a threshold
that passes on a 4070 Ti and fails in CI is a threshold nobody trusts.

    pytest tests/test_bench_serving.py -m bench -s          # mock runner, no GPU
    BENCH_MODEL_ID=Qwen/Qwen3-0.6B pytest ... -m bench -s   # real weights
    MODEL_SERVER_URL=http://127.0.0.1:8000 pytest ... -m bench -s   # a deployed pod

The default is the MOCK runner, deliberately. The scheduler-level claims above
are true statements about batching.py, not about Qwen -- MockRunner drives the
same Scheduler and the same run_loop, so they can be graded in ~20 seconds on a
laptop with no GPU and no weights. Point BENCH_MODEL_ID at real weights when you
want real numbers rather than a regression gate.

Knobs: BENCH_CONCURRENCY (default 1,2,4,8,16), BENCH_MAX_TOKENS (32),
BENCH_MAX_BATCH_SIZE (8), BENCH_REQUESTS_PER_WORKER (4), BENCH_MAX_REQUESTS (64),
BENCH_JSON (write the raw table somewhere), and the opt-in budgets
BENCH_MAX_TTFT_MS / BENCH_MAX_P99_MS / BENCH_MIN_OUTPUT_TOK_S.
"""

import asyncio
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field

import httpx
import pytest

from conftest import metric_value, serve, wait_until_ready

pytestmark = pytest.mark.bench

MODEL_ID = os.environ.get("BENCH_MODEL_ID", "mock/bench")
EXTERNAL_URL = os.environ.get("MODEL_SERVER_URL")
MAX_BATCH_SIZE = int(os.environ.get("BENCH_MAX_BATCH_SIZE", "8"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "32"))
CONCURRENCY = [int(c) for c in os.environ.get("BENCH_CONCURRENCY", "1,2,4,8,16").split(",")]
REQUESTS_PER_WORKER = int(os.environ.get("BENCH_REQUESTS_PER_WORKER", "4"))
MAX_REQUESTS = int(os.environ.get("BENCH_MAX_REQUESTS", "64"))
READY_TIMEOUT = float(os.environ.get("BENCH_LOAD_TIMEOUT", "600"))

PROMPT = os.environ.get(
    "BENCH_PROMPT", "Explain what a Kubernetes operator does, in a few sentences."
)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


@dataclass
class RequestStat:
    ttft: float
    latency: float
    chunks: int
    itls: list[float] = field(default_factory=list)

    @property
    def tpot(self) -> float:
        return statistics.fmean(self.itls) if self.itls else 0.0


@dataclass
class RunStats:
    """One closed-loop run at a fixed concurrency."""

    concurrency: int
    requests: int
    duration: float
    output_chunks: int
    output_tok_s: float
    request_s: float
    ttft_p50: float
    ttft_p95: float
    tpot_mean: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. n is small here; interpolation would invent data."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


async def _one_request(client: httpx.AsyncClient, max_tokens: int) -> RequestStat:
    """One streaming completion, timed at the SSE-chunk level.

    Streaming rather than a single POST because TTFT is the whole point: a
    non-streaming request cannot distinguish "queued for 3 seconds then decoded
    fast" from "started instantly and decoded slowly", and those two services
    feel completely different to a user.

    Chunks, not tokens: the engine yields only when the decoded text grew, so a
    CJK character spanning 2-3 BPE tokens produces one chunk. On ASCII prompts
    chunks == tokens; on multibyte output this undercounts, which makes tok/s a
    lower bound rather than a fiction.
    """
    body = {"prompt": PROMPT, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    start = time.perf_counter()
    ttft, previous, itls, chunks = None, None, [], 0

    async with client.stream("POST", "/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            if not json.loads(payload)["choices"][0]["text"]:
                continue  # the terminal finish_reason chunk carries no text
            now = time.perf_counter()
            chunks += 1
            if ttft is None:
                ttft = now - start
            else:
                itls.append(now - previous)
            previous = now

    latency = time.perf_counter() - start
    return RequestStat(ttft=ttft if ttft is not None else latency, latency=latency,
                       chunks=chunks, itls=itls)


async def _run_load(base_url: str, concurrency: int, total: int, max_tokens: int) -> RunStats:
    """Closed loop: `concurrency` workers, each looping until `total` are done.

    Closed rather than open (a fixed arrival rate) because it cannot overload
    itself into meaningless numbers: the offered load is exactly `concurrency`
    in flight, which is the variable being swept. An open loop measures where the
    service collapses -- a different, later experiment.
    """
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(base_url=base_url, timeout=600, limits=limits) as client:
        remaining = total
        lock = asyncio.Lock()
        stats: list[RequestStat] = []

        async def worker():
            nonlocal remaining
            while True:
                async with lock:
                    if remaining <= 0:
                        return
                    remaining -= 1
                stats.append(await _one_request(client, max_tokens))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        duration = time.perf_counter() - start

    chunks = sum(s.chunks for s in stats)
    return RunStats(
        concurrency=concurrency,
        requests=len(stats),
        duration=duration,
        output_chunks=chunks,
        output_tok_s=chunks / duration,
        request_s=len(stats) / duration,
        ttft_p50=_pct([s.ttft for s in stats], 50) * 1000,
        ttft_p95=_pct([s.ttft for s in stats], 95) * 1000,
        tpot_mean=statistics.fmean([s.tpot for s in stats if s.itls]) * 1000,
        e2e_p50=_pct([s.latency for s in stats], 50) * 1000,
        e2e_p95=_pct([s.latency for s in stats], 95) * 1000,
        e2e_p99=_pct([s.latency for s in stats], 99) * 1000,
    )


def run_load(base_url: str, concurrency: int, total: int | None = None,
             max_tokens: int = MAX_TOKENS) -> RunStats:
    if total is None:
        total = min(MAX_REQUESTS, max(MAX_BATCH_SIZE, REQUESTS_PER_WORKER * concurrency))
    return asyncio.run(_run_load(base_url, concurrency, total, max_tokens))


# ---------------------------------------------------------------------------
# the service under test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def service():
    if EXTERNAL_URL:
        wait_until_ready(EXTERNAL_URL, READY_TIMEOUT)
        yield EXTERNAL_URL
        return

    if not MODEL_ID.startswith("mock/"):
        pytest.importorskip("torch", reason="real-weight benchmarking needs torch")
        pytest.importorskip("transformers")

    from main import Settings, create_app

    app = create_app(
        Settings(
            model_name=MODEL_ID,
            model_id=MODEL_ID,
            load_time_seconds=0,
            max_batch_size=MAX_BATCH_SIZE,
            enable_thinking=False,
        )
    )
    with serve(app, ready_timeout=READY_TIMEOUT) as base_url:
        yield base_url


@pytest.fixture(scope="module")
def sweep(service):
    """The concurrency sweep, run ONCE and shared by every assertion below.

    Warmup first, discarded: on the real runner the first request pays for CUDA
    context creation, kernel autotuning and the first cache allocation -- tens of
    seconds that belong to no percentile.
    """
    run_load(service, concurrency=2, total=2, max_tokens=8)

    results: dict[int, RunStats] = {}
    print(f"\n=== serving benchmark: {MODEL_ID} ===")
    print(f"max_batch_size={MAX_BATCH_SIZE} max_tokens={MAX_TOKENS} prompt={PROMPT!r}")
    print(f"{'conc':>4} {'reqs':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'TPOT':>7} "
          f"{'E2E p50':>9} {'E2E p99':>9} {'tok/s':>8} {'req/s':>7}")
    for concurrency in CONCURRENCY:
        stats = results[concurrency] = run_load(service, concurrency)
        print(f"{stats.concurrency:>4} {stats.requests:>5} {stats.ttft_p50:>8.1f}ms "
              f"{stats.ttft_p95:>8.1f}ms {stats.tpot_mean:>6.1f}ms {stats.e2e_p50:>8.1f}ms "
              f"{stats.e2e_p99:>8.1f}ms {stats.output_tok_s:>8.1f} {stats.request_s:>7.2f}")

    if os.environ.get("BENCH_JSON"):
        # Raw rows rather than a verdict: a benchmark is only a regression gate
        # once you can diff today's run against last week's on the same box.
        with open(os.environ["BENCH_JSON"], "w") as handle:
            json.dump(
                {"model": MODEL_ID, "max_batch_size": MAX_BATCH_SIZE,
                 "max_tokens": MAX_TOKENS,
                 "runs": [asdict(r) for r in results.values()]},
                handle, indent=2,
            )
    return results


def _at(sweep: dict[int, RunStats], concurrency: int) -> RunStats:
    if concurrency not in sweep:
        pytest.skip(f"concurrency {concurrency} not in BENCH_CONCURRENCY={CONCURRENCY}")
    return sweep[concurrency]


# --- 1. batching earns its complexity --------------------------------------


def test_batching_lifts_output_throughput(sweep):
    """A full batch must produce far more tokens per second than one request does.

    This is the entire justification for continuous batching, stated as a
    number. Decode is memory-bandwidth bound: a batch of 8 reads the same
    weights once and serves 8 rows, so throughput should climb close to
    linearly until the batch is full. If this ratio ever falls back toward 1.0,
    requests are being serialized somewhere -- a lock, an await in the wrong
    place, a scheduler admitting one row at a time -- and no unit test would say
    so, because each layer is still individually correct.

    2x is a floor, not a target: the mock reaches ~8x and a GPU reaches 4-6x.
    """
    floor = float(os.environ.get("BENCH_MIN_BATCH_SPEEDUP", "2.0"))
    serial, batched = _at(sweep, 1), _at(sweep, MAX_BATCH_SIZE)

    speedup = batched.output_tok_s / serial.output_tok_s

    print(f"\nbatching speedup at C={MAX_BATCH_SIZE}: {speedup:.1f}x "
          f"({serial.output_tok_s:.0f} -> {batched.output_tok_s:.0f} tok/s)")
    assert speedup >= floor, (
        f"batching bought only {speedup:.2f}x; requests look serialized"
    )


def test_latency_stays_sublinear_in_concurrency(sweep):
    """Eight at once must not take eight times as long each.

    The counterpart to the throughput test, and the one that catches a service
    that "scales" by simply queueing: total throughput can look fine while every
    individual user waits proportionally longer. Under real batching the rows
    share forward passes, so per-request latency should barely move until the
    batch is full.
    """
    ceiling = float(os.environ.get("BENCH_MAX_LATENCY_INFLATION", "2.0"))
    serial, batched = _at(sweep, 1), _at(sweep, MAX_BATCH_SIZE)

    inflation = batched.e2e_p50 / serial.e2e_p50

    print(f"\nlatency inflation at C={MAX_BATCH_SIZE}: {inflation:.2f}x "
          f"({serial.e2e_p50:.0f} -> {batched.e2e_p50:.0f} ms p50); "
          f"serialized would be {MAX_BATCH_SIZE}x")
    assert inflation <= ceiling, (
        f"p50 latency grew {inflation:.1f}x for {MAX_BATCH_SIZE} concurrent requests"
    )


def test_the_tail_stays_bounded_under_uniform_load(sweep):
    """Identical requests must finish in roughly identical time.

    FIFO admission is what bounds the tail: `break, not skip-ahead` in
    Scheduler.schedule. Packing a smaller request past a queued one improves the
    throughput graph and starves the request at the head -- which shows up
    nowhere except in p99/p50. Uniform load makes any spread here a scheduling
    artifact rather than a workload property.
    """
    ratio_max = float(os.environ.get("BENCH_MAX_TAIL_RATIO", "3.0"))
    stats = _at(sweep, MAX_BATCH_SIZE)

    ratio = stats.e2e_p99 / stats.e2e_p50

    print(f"\ntail at C={MAX_BATCH_SIZE}: p50 {stats.e2e_p50:.0f}ms "
          f"p95 {stats.e2e_p95:.0f}ms p99 {stats.e2e_p99:.0f}ms ({ratio:.2f}x)")
    assert ratio <= ratio_max, f"p99 is {ratio:.1f}x p50 on identical requests"


def test_offering_more_than_the_batch_does_not_raise_throughput(sweep):
    """Past the batch limit, extra concurrency buys queue depth, not capacity.

    The knee is the useful output of this whole sweep: it is the number an HPA
    target and a capacity plan are set from. Asserting the SHAPE (throughput
    flattens rather than climbing) rather than a value keeps it portable.
    """
    over = MAX_BATCH_SIZE * 2
    if over not in sweep:
        pytest.skip(f"no C={over} in the sweep")
    batched, overloaded = _at(sweep, MAX_BATCH_SIZE), sweep[over]

    gain = overloaded.output_tok_s / batched.output_tok_s

    print(f"\nC={over} vs C={MAX_BATCH_SIZE}: {gain:.2f}x throughput, "
          f"TTFT p95 {batched.ttft_p95:.0f} -> {overloaded.ttft_p95:.0f}ms")
    assert gain <= 1.6, (
        f"throughput rose {gain:.2f}x past max_batch_size={MAX_BATCH_SIZE}; "
        "the admission limit is not binding"
    )
    assert overloaded.ttft_p95 >= batched.ttft_p95, (
        "queueing beyond the batch limit must show up as TTFT, not vanish"
    )


# --- 2. the metric an autoscaler acts on is real ---------------------------


def test_queue_depth_is_visible_while_the_batch_is_full(service):
    """Overload the service and watch /metrics from outside.

    `vllm:num_requests_waiting` is what an HPA scales on, so it has to be a
    measurement and not a decoration. Three claims, all of them load-bearing for
    autoscaling: running never exceeds max_batch_size (the limit binds), running
    actually REACHES it (the batch fills rather than trickling), and waiting goes
    above zero (backpressure is observable rather than hidden in a socket
    buffer). Sampled from a separate thread during the load, because after it
    finishes both gauges are zero and prove nothing.
    """
    samples: list[tuple[int, int]] = []
    stop = threading.Event()

    def scrape():
        with httpx.Client(base_url=service, timeout=10) as http:
            while not stop.is_set():
                text = http.get("/metrics").text
                samples.append((
                    metric_value(text, "vllm:num_requests_running"),
                    metric_value(text, "vllm:num_requests_waiting"),
                ))
                time.sleep(0.02)

    scraper = threading.Thread(target=scrape, daemon=True)
    scraper.start()
    try:
        run_load(service, concurrency=MAX_BATCH_SIZE * 2, total=MAX_BATCH_SIZE * 4)
    finally:
        stop.set()
        scraper.join(timeout=5)

    peak_running = max(running for running, _ in samples)
    peak_waiting = max(waiting for _, waiting in samples)

    print(f"\npeak running={peak_running} (limit {MAX_BATCH_SIZE}) waiting={peak_waiting}")
    assert peak_running <= MAX_BATCH_SIZE, "admission let more rows in than the batch allows"
    assert peak_running == MAX_BATCH_SIZE, "the batch never filled; admission is too timid"
    assert peak_waiting > 0, "overload produced no visible queue for an autoscaler to see"

    with httpx.Client(base_url=service, timeout=10) as http:
        text = http.get("/metrics").text
    assert metric_value(text, "vllm:num_requests_running") == 0
    assert metric_value(text, "vllm:num_requests_waiting") == 0


# --- 3. opt-in absolute budgets (CI gate on known hardware) ----------------


def test_meets_the_configured_service_level(sweep):
    """Hard numbers, only when someone has committed to hard numbers.

    Relative assertions catch regressions in the code; they cannot catch a
    machine that is simply too slow, and a threshold hard-coded here would fail
    on every box that is not the author's. So these live in env:

        BENCH_MAX_TTFT_MS=500 BENCH_MAX_P99_MS=4000 BENCH_MIN_OUTPUT_TOK_S=200 \\
        BENCH_MODEL_ID=Qwen/Qwen3-0.6B pytest tests/test_bench_serving.py -m bench -s
    """
    budgets = {
        "ttft_p95": os.environ.get("BENCH_MAX_TTFT_MS"),
        "e2e_p99": os.environ.get("BENCH_MAX_P99_MS"),
        "output_tok_s": os.environ.get("BENCH_MIN_OUTPUT_TOK_S"),
    }
    if not any(budgets.values()):
        pytest.skip("no absolute budget set (BENCH_MAX_TTFT_MS / BENCH_MAX_P99_MS / "
                    "BENCH_MIN_OUTPUT_TOK_S)")

    stats = _at(sweep, MAX_BATCH_SIZE)

    if budgets["ttft_p95"]:
        assert stats.ttft_p95 <= float(budgets["ttft_p95"]), f"TTFT p95 {stats.ttft_p95:.0f}ms"
    if budgets["e2e_p99"]:
        assert stats.e2e_p99 <= float(budgets["e2e_p99"]), f"E2E p99 {stats.e2e_p99:.0f}ms"
    if budgets["output_tok_s"]:
        assert stats.output_tok_s >= float(budgets["output_tok_s"]), (
            f"throughput {stats.output_tok_s:.0f} tok/s"
        )
