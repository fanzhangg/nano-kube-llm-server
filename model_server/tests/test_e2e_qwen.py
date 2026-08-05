"""End-to-end contract check of the Qwen3 service: real weights, real HTTP.

**Not an eval, and not a quality check.** Nothing here judges whether an answer
is good, true, or on-topic. That is a model question; it needs a dataset and a
scored metric, and a handful of hand-picked prompts would only measure how well
the prompts were picked. What this file grades is the DATA CONTRACT the service
promises its clients, with real weights behind it and especially while requests
are sharing a batch:

    every response is well-formed and parses as an OpenAI Completion
    usage adds up, and describes THAT request's prompt, not its neighbour's
    finish_reason names the bound that actually stopped the sequence
    each SSE stream carries one id from first chunk to [DONE]

Why real weights are needed for a contract test at all: every layer's unit test
is already green with a stub. test_main.py drives the HTTP layer over
MockRunner, whose "tokens" are characters -- so prompt_tokens == len(prompt) and
no chat template, no vocabulary, no EOS token is involved. The accounting a
client actually receives (a templated prompt count, a real stop token, ragged
rows padded into one forward pass) only exists once QwenRunner is in the path.
That is the seam this file covers, and it is a shallow one on purpose: the
assertions are about shape, counts and identity, all of which are true
statements about the SERVICE regardless of which model is loaded behind it.

Opt-in, because it loads ~1.2GB:

    RUN_E2E=1 pytest tests/test_e2e_qwen.py -m e2e -v           # in-process server
    RUN_E2E=1 MODEL_SERVER_URL=http://localhost:8000 pytest ... # a deployed pod

The second form is the same file pointed at a real ModelServer -- port-forward a
pod and every assertion below becomes a cluster acceptance test:

    kubectl port-forward pod/<modelserver-pod> 8000:8000

Forward a POD, not the Service, whenever replicas > 1: the last test reads
/metrics and compares it against a request it just made, and a Service would
round-robin the scrape onto a pod that never saw that request.

Knobs:

    E2E_MODEL_ID        default Qwen/Qwen3-0.6B
    E2E_LOAD_TIMEOUT    seconds to wait for /health 200 (default 600)
    E2E_ALLOW_DOWNLOAD  1 to permit a hub download instead of skipping
"""

import concurrent.futures
import json
import os
import time

import httpx
import pytest
from openai import OpenAI

from conftest import metric_value, serve, wait_until_ready

pytestmark = pytest.mark.e2e

MODEL_ID = os.environ.get("E2E_MODEL_ID", "Qwen/Qwen3-0.6B")
EXTERNAL_URL = os.environ.get("MODEL_SERVER_URL")
LOAD_TIMEOUT = float(os.environ.get("E2E_LOAD_TIMEOUT", "600"))

# Inputs, not questions with answers. What matters is that they tokenize to
# DIFFERENT lengths: prompt_tokens is then a fingerprint, and a response that
# ends up on the wrong client is visible as a number that belongs to another
# request. Content is irrelevant to every assertion below.
PROMPTS = [
    "Name a color.",
    "Describe the ocean in one sentence.",
    "List three fruits, separated by commas, and nothing else.",
    "Here is a list you should ignore: " + "apple, banana, cherry, " * 30 + "Say hello.",
]
CJK_PROMPT = "用一句话介绍北京。"


def _weights_are_cached(model_id: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def service():
    """Base URL of a Qwen3 service that has finished loading.

    Module-scoped: loading the weights is the expensive part, and every test here
    leaves the server as it found it.
    """
    if not os.environ.get("RUN_E2E"):
        pytest.skip("e2e is opt-in: set RUN_E2E=1")

    if EXTERNAL_URL:
        # No lifecycle to manage -- something else (kubectl port-forward, a
        # Service, docker run) owns that process. Still wait for readiness, so a
        # pod caught mid-load fails on a clear timeout instead of on 503s.
        wait_until_ready(EXTERNAL_URL, LOAD_TIMEOUT)
        yield EXTERNAL_URL
        return

    pytest.importorskip("torch", reason="the real service needs torch")
    pytest.importorskip("transformers", reason="the real service needs transformers")
    if not _weights_are_cached(MODEL_ID) and not os.environ.get("E2E_ALLOW_DOWNLOAD"):
        pytest.skip(
            f"{MODEL_ID} is not in the local HF cache; "
            f"set E2E_ALLOW_DOWNLOAD=1 to fetch ~1.2GB"
        )

    from main import Settings, create_app

    app = create_app(
        Settings(
            model_name=MODEL_ID,
            model_id=MODEL_ID,
            load_time_seconds=0,  # the real load is the wait; no artificial window
            max_batch_size=8,
            enable_thinking=False,
        )
    )
    with serve(app, ready_timeout=LOAD_TIMEOUT) as base_url:
        yield base_url


@pytest.fixture(scope="module")
def client(service):
    # The official SDK rather than raw httpx: it parses each response into its
    # own Completion model, so a missing or misshapen field raises inside the SDK
    # before any assertion here runs. Half the contract is checked for free.
    return OpenAI(base_url=f"{service}/v1", api_key="not-needed", timeout=600, max_retries=0)


@pytest.fixture(scope="module")
def prompt_token_count():
    """len(prompt_ids) as the SERVER should compute it, from the same tokenizer.

    An independent second opinion on usage.prompt_tokens -- which is the field
    that identifies which request a response belongs to.
    """
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    def count(prompt: str) -> int:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(templated).input_ids)

    return count


def greedy(client, prompt: str, max_tokens: int = 16, **kwargs):
    """One completion at temperature=0.

    Greedy not for reproducible text -- no assertion here looks at the text --
    but so that token COUNTS do not wander between runs of the same test.
    """
    return client.completions.create(
        model=MODEL_ID, prompt=prompt, max_tokens=max_tokens, temperature=0, **kwargs
    )


def assert_shape(completion, max_tokens: int) -> None:
    """The invariants every completion owes its client, batched or not."""
    assert completion.id.startswith("cmpl-")
    assert completion.object == "text_completion"
    assert completion.model
    assert len(completion.choices) == 1
    assert completion.choices[0].index == 0
    assert isinstance(completion.choices[0].text, str)
    assert completion.choices[0].finish_reason in {"stop", "length"}
    assert completion.usage.prompt_tokens > 0
    assert 0 <= completion.usage.completion_tokens <= max_tokens
    assert (
        completion.usage.total_tokens
        == completion.usage.prompt_tokens + completion.usage.completion_tokens
    )
    # The two are the same event seen from two sides; either without the other
    # means the number a client bills on and the reason it stopped disagree.
    stopped_early = completion.usage.completion_tokens < max_tokens
    assert stopped_early == (completion.choices[0].finish_reason == "stop")


# --- 1. one request at a time ----------------------------------------------


def test_health_reports_the_loaded_model(service):
    body = httpx.get(f"{service}/health", timeout=30).json()

    assert body["status"] == "ok"
    assert body["model"]  # the name the controller wrote into MODEL_NAME


def test_a_single_completion_satisfies_the_contract(client):
    completion = greedy(client, PROMPTS[0], max_tokens=16)

    assert_shape(completion, max_tokens=16)
    assert completion.choices[0].text.strip()  # real weights, real tokens


def test_usage_counts_the_templated_prompt_not_the_raw_string(client, prompt_token_count):
    """prompt_tokens must count the chat control tokens the model actually saw.

    Also the objective grader for QwenRunner.encode: drop the chat template and
    Qwen3 continues the prompt instead of responding to it. Judging the text
    would catch that only by luck; the token count catches it as an exact
    shortfall. Note this claim cannot even be stated against MockRunner, where a
    token is a character and there is no template at all.
    """
    completion = greedy(client, PROMPTS[0], max_tokens=8)

    assert completion.usage.prompt_tokens == prompt_token_count(PROMPTS[0])


def test_the_length_limit_is_exact(client):
    """A budget of N tokens means N tokens, and finish_reason names the bound."""
    completion = greedy(client, "Write a long story about a robot.", max_tokens=16)

    assert completion.choices[0].finish_reason == "length"
    assert completion.usage.completion_tokens == 16


def test_a_sequence_that_ends_on_its_own_reports_stop(client):
    """Given room, a stop token must both end the sequence and be reported.

    Not a claim about the answer -- a claim that the stop-token plumbing is
    connected. Qwen3 has TWO stop tokens and only <|im_end|> is
    tokenizer.eos_token_id; miss <|endoftext|> and the sequence runs to
    max_new_tokens and reports "length" for a request that had already finished.
    The end-to-end grader for QwenRunner._resolve_eos_ids.
    """
    completion = greedy(client, PROMPTS[0], max_tokens=128)

    assert completion.usage.completion_tokens < 128, (
        "no stop token in 128 tokens -- eos_token_ids is likely incomplete"
    )
    assert completion.choices[0].finish_reason == "stop"


def test_zero_max_tokens_generates_nothing(client):
    """A legal OpenAI request that must be resolved at admission, not after a tick."""
    completion = greedy(client, PROMPTS[0], max_tokens=0)

    assert completion.choices[0].text == ""
    assert completion.usage.completion_tokens == 0
    assert completion.choices[0].finish_reason == "length"


# --- 2. the same contract while requests share a batch ---------------------


def test_every_response_in_a_batch_is_well_formed(client):
    """Four ragged prompts in flight at once; four intact responses out.

    Ragged deliberately -- the longest is ~250 tokens more than the shortest --
    because that is what forces left-padding, and padding is what a batch can
    corrupt. This is the shape half of the check.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        results = list(pool.map(lambda p: greedy(client, p, max_tokens=12), PROMPTS))

    for completion in results:
        assert_shape(completion, max_tokens=12)
    assert len({c.id for c in results}) == len(results), "two responses shared an id"


def test_a_batched_response_reports_its_own_prompt_not_a_neighbours(
    client, prompt_token_count
):
    """The identity half: each response's accounting must belong to ITS request.

    Fan-out from one batched forward pass back to N HTTP responses is
    Sequence.queue and nothing else. A row-indexing slip in postprocess or in
    sampling hands B's tokens to A's client -- coherent text, 200 OK, wrong
    recipient. Prompt lengths differ by hundreds of tokens here, so a swap is a
    number that cannot be explained away, and no judgement about the text is
    needed to see it.
    """
    expected = {prompt: prompt_token_count(prompt) for prompt in PROMPTS}
    assert len(set(expected.values())) == len(PROMPTS), "prompts must be distinguishable"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        results = list(pool.map(lambda p: greedy(client, p, max_tokens=12), PROMPTS))

    for prompt, completion in zip(PROMPTS, results):
        assert completion.usage.prompt_tokens == expected[prompt], (
            f"{prompt[:40]!r} came back with prompt_tokens="
            f"{completion.usage.prompt_tokens}, expected {expected[prompt]} -- "
            "responses are crossing over between batched rows"
        )


def test_a_batch_that_overflows_the_limit_still_answers_everyone(client):
    """More concurrent requests than max_batch_size: queued, not dropped.

    Admission is a capacity limit, not an error path -- backpressure is the
    queue. The contract a client sees must be identical whether it was admitted
    immediately or waited: a full, well-formed response, never a 503 or a
    truncated body.
    """
    over_capacity = 12  # the fixture's max_batch_size is 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=over_capacity) as pool:
        results = list(
            pool.map(lambda _: greedy(client, PROMPTS[0], max_tokens=8), range(over_capacity))
        )

    assert len(results) == over_capacity
    for completion in results:
        assert_shape(completion, max_tokens=8)


# --- 3. the streaming contract ---------------------------------------------


def test_the_sse_stream_is_framed_correctly(service):
    """Raw SSE, read off the socket: framing is the part the SDK hides.

    One id for every chunk in a stream (a client correlates on it), the OpenAI
    object type on each, a terminal chunk carrying finish_reason, and the
    literal [DONE] sentinel last. Read with httpx rather than the SDK because
    the SDK parses [DONE] away and would let a malformed terminator through.
    """
    max_tokens = 8
    with httpx.Client(base_url=service, timeout=120) as http:
        with http.stream(
            "POST",
            "/v1/completions",
            json={"prompt": PROMPTS[0], "max_tokens": max_tokens,
                  "stream": True, "temperature": 0},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payloads = [
                line[len("data: "):] for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert len({c["id"] for c in chunks}) == 1, "chunks of one stream must share an id"
    assert all(c["object"] == "text_completion" for c in chunks)
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    # Text-bearing chunks are tokens (minus any that completed a multibyte
    # character), so the budget bounds them.
    assert 0 < sum(1 for c in chunks if c["choices"][0]["text"]) <= max_tokens


def test_a_stream_that_ends_on_eos_reports_stop(service):
    """Streaming and buffered must not disagree about why a sequence ended.

    Same request, same greedy path, two transports: whatever finish_reason the
    buffered response gives, the stream's terminal chunk owes the client the
    same one. This started life as an xfail -- main._stream hardcoded "length"
    on its terminal chunk, so every stream claimed truncation and clients that
    continue on "length" would keep asking a finished answer for more. The fix
    is engines.StreamChunk; this is what holds it.
    """
    body = {"prompt": PROMPTS[0], "max_tokens": 128, "temperature": 0}
    with httpx.Client(base_url=service, timeout=300) as http:
        buffered = http.post("/v1/completions", json=body).json()
        with http.stream("POST", "/v1/completions", json={**body, "stream": True}) as resp:
            payloads = [
                line[len("data: "):] for line in resp.iter_lines()
                if line.startswith("data: ")
            ]

    streamed_finish = json.loads(payloads[-2])["choices"][0]["finish_reason"]
    assert streamed_finish == buffered["choices"][0]["finish_reason"]


def test_multibyte_output_is_never_split_across_chunks(client):
    """A CJK character is 2-3 BPE tokens; decoding each token alone yields U+FFFD.

    The engine decodes the whole output each round and yields only the new
    suffix, so an incomplete character contributes nothing that round and
    appears once complete. This is the grader for that, and it fails loudly
    against the naive per-token decode -- which looks perfect in every ASCII
    test in this repo. Skips rather than passes vacuously when the model
    happened to answer in ASCII.
    """
    text = "".join(
        event.choices[0].text
        for event in greedy(client, CJK_PROMPT, max_tokens=24, stream=True)
    )

    if not any(ord(char) > 127 for char in text):
        pytest.skip(f"model answered {CJK_PROMPT!r} in ASCII; nothing multibyte to check")
    assert "�" not in text


# --- 4. a bad request costs one request ------------------------------------
#
# The blast-radius claim, stated where it can actually be tested: with real
# weights, a real scheduler, and real neighbours in the batch. test_main.py
# grades the same rules against MockRunner, which never samples -- so it cannot
# show that a rejected request would otherwise have reached torch.


@pytest.mark.parametrize(
    "body,param",
    [
        ({"prompt": "hi", "top_k": 2.5}, "top_k"),
        ({"prompt": "hi", "max_tokens": -1}, "max_tokens"),
        ({"prompt": "hi", "temperature": -0.5}, "temperature"),
    ],
)
def test_an_invalid_request_is_rejected_with_the_openai_error_envelope(service, body, param):
    """400 and {"error": {...}}, not 200 with a finish_reason outside the enum.

    Raw httpx, not the SDK, because the SDK raises BadRequestError on a 400 and
    the point here is the BODY: `type` and `param` are what a client displays.
    """
    response = httpx.post(f"{service}/v1/completions", json=body, timeout=60)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param


def test_rejected_requests_do_not_disturb_the_requests_around_them(client, service):
    """Bad requests interleaved with good ones, all in flight together.

    This is the regression that started it: {"top_k": "5"} raised inside a
    forward pass shared with everyone else, fail_all took the whole batch down,
    the loop died, and /health went on reporting 200 while every later request
    hung forever. Every part of that is asserted here -- the good responses
    survive intact, the server is still ready, and the batch drains to empty.
    """
    def bad():
        return httpx.post(
            f"{service}/v1/completions",
            json={"prompt": "hi", "max_tokens": 8, "top_k": 2.5},
            timeout=60,
        ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        good = [pool.submit(greedy, client, prompt, 12) for prompt in PROMPTS]
        rejected = [pool.submit(bad) for _ in range(4)]
        completions = [future.result() for future in good]
        statuses = [future.result() for future in rejected]

    assert statuses == [400] * 4
    for completion in completions:
        assert_shape(completion, max_tokens=12)

    assert httpx.get(f"{service}/health", timeout=30).status_code == 200
    metrics = httpx.get(f"{service}/metrics", timeout=30).text
    assert metric_value(metrics, "vllm:num_requests_running") == 0
    assert metric_value(metrics, "vllm:num_requests_waiting") == 0


# --- 5. a slot released is a slot returned ---------------------------------


def test_abandoning_a_stream_frees_the_batch_slot(service):
    """Hang up mid-stream; the slot must not keep generating for nobody.

    Not the output contract, but the reason a batch stays available to honour
    it: a leak here is invisible until the batch is permanently full of ghosts
    and every later request queues forever. /metrics is the grader because that
    is the same signal the autoscaler reads.
    """
    with httpx.Client(base_url=service, timeout=60) as http:
        with http.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "Write a very long story.", "max_tokens": 512, "stream": True},
        ) as response:
            for i, _ in enumerate(response.iter_lines()):
                if i >= 3:
                    break  # leaving the context manager closes the connection

        for _ in range(200):  # the abort lands on the next tick, not instantly
            running = metric_value(http.get("/metrics").text, "vllm:num_requests_running")
            if running == 0:
                break
            time.sleep(0.1)

        assert running == 0, "the abandoned sequence still holds a batch slot"
