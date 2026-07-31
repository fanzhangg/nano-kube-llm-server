"""Grader for the Milestone 3 Qwen3 skeleton (docs/qwen-tutorial.md).

These tests define the contract for every TODO in qwen.py. Work through the
skeleton until `pytest test_qwen.py -v` is green -- each test names the property
it checks.

Deliberately cheap: no 1.2GB checkpoint and no GPU. The tokenizer is real (a few
MB, and the chat template is the thing under test), but the model is either a
stub returning canned ids or a ~5M-param randomly-initialised Qwen3. Random
weights emit gibberish, which is fine -- we are grading the *contract* (is the
prompt sliced off? are the counts honest? is finish_reason right?), never the
quality of the text.
"""

import pytest

torch = pytest.importorskip("torch", reason="milestone 3 needs torch")
pytest.importorskip("transformers", reason="milestone 3 needs transformers")

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from qwen import (  # noqa: E402
    Generation,
    build_prompt,
    complete,
    complete_stream,
    pick_dtype,
    sampling_kwargs,
)

MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    """The real Qwen3 tokenizer -- tokenizer files only, no weights (~11MB)."""
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # offline and not cached
        pytest.skip(f"cannot load {MODEL_ID} tokenizer: {exc}")


class RecordingTokenizer:
    """Captures whatever build_prompt passes to apply_chat_template."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "<rendered>"


class StubModel:
    """A model whose generate() returns ids we choose, so the slicing, counting
    and finish_reason logic can be graded exactly rather than statistically."""

    def __init__(self, output_ids):
        self.output_ids = torch.tensor([output_ids])
        self.device = torch.device("cpu")
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.output_ids


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    """A real Qwen3 with random weights: same architecture and API as the 0.6B,
    ~5M params so it runs on CPU in well under a second."""
    cfg = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


class TestPickDtype:
    def test_cuda_gets_bfloat16(self):
        assert pick_dtype("cuda") is torch.bfloat16
        assert pick_dtype("cuda:0") is torch.bfloat16

    def test_cpu_gets_float32(self):
        # bf16 on CPU mostly hits slow emulated kernels -- it is a pessimisation.
        assert pick_dtype("cpu") is torch.float32


class TestSamplingKwargs:
    def test_zero_temperature_is_greedy(self):
        # OpenAI spells "deterministic" as temperature=0; HF spells it
        # do_sample=False and rejects temperature=0 outright.
        assert sampling_kwargs(0.0, None) == {"do_sample": False}

    def test_greedy_carries_no_sampling_knobs(self):
        # Passing top_k/temperature alongside do_sample=False makes generate()
        # warn that they cannot take effect.
        assert sampling_kwargs(0.0, 20) == {"do_sample": False}
        assert sampling_kwargs(-1.0, 50) == {"do_sample": False}

    def test_positive_temperature_samples(self):
        assert sampling_kwargs(0.7, None) == {"do_sample": True, "temperature": 0.7}

    def test_top_k_included_when_positive(self):
        assert sampling_kwargs(0.7, 20) == {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 20,
        }

    @pytest.mark.parametrize("bad", [None, 0, -5])
    def test_non_positive_top_k_omitted(self, bad):
        # top_k is an optional vLLM extension; absent or nonsensical means "unset",
        # not "top_k=0" (which HF would read as a hard filter to zero tokens).
        assert "top_k" not in sampling_kwargs(0.7, bad)


class TestBuildPrompt:
    def test_wraps_prompt_as_a_user_message(self):
        tok = RecordingTokenizer()
        build_prompt(tok, "hello")
        (messages, _), = tok.calls
        assert messages == [{"role": "user", "content": "hello"}]

    def test_returns_a_string_not_ids(self):
        tok = RecordingTokenizer()
        assert build_prompt(tok, "hello") == "<rendered>"
        _, kwargs = tok.calls[0]
        assert kwargs["tokenize"] is False

    def test_adds_generation_prompt(self):
        # Without it the rendered text ends after the user turn, and the model has
        # no assistant turn to speak in -- it continues the *user* instead.
        tok = RecordingTokenizer()
        build_prompt(tok, "hello")
        assert tok.calls[0][1]["add_generation_prompt"] is True

    @pytest.mark.parametrize("thinking", [True, False])
    def test_thinking_flag_passed_through(self, thinking):
        tok = RecordingTokenizer()
        build_prompt(tok, "hello", enable_thinking=thinking)
        assert tok.calls[0][1]["enable_thinking"] is thinking

    def test_real_template_defaults_to_no_thinking(self, tokenizer):
        # enable_thinking=False does not delete the block, it prefills an EMPTY
        # one so the model treats reasoning as already done.
        text = build_prompt(tokenizer, "hello")
        assert "<|im_start|>assistant" in text
        assert "<think>" in text and "</think>" in text

    def test_real_template_with_thinking_leaves_block_open(self, tokenizer):
        text = build_prompt(tokenizer, "hello", enable_thinking=True)
        assert "</think>" not in text


class TestComplete:
    def _run(self, tokenizer, output_ids, max_new_tokens=8):
        model = StubModel(output_ids)
        result = complete(model, tokenizer, "hi", max_new_tokens, temperature=0.0)
        return model, result

    def test_returns_a_generation(self, tokenizer):
        _, result = self._run(tokenizer, [1, 2, 3, 4, 5])
        assert isinstance(result, Generation)

    def test_prompt_is_sliced_off(self, tokenizer):
        # The classic bug: generate() returns prompt + completion concatenated, so
        # skipping the slice echoes the whole templated prompt back to the caller.
        n_prompt = len(tokenizer(build_prompt(tokenizer, "hi")).input_ids)
        tail = tokenizer.encode(" world")
        _, result = self._run(tokenizer, list(range(n_prompt)) + tail)
        assert result.text == tokenizer.decode(tail, skip_special_tokens=True)

    def test_special_tokens_are_not_leaked(self, tokenizer):
        n_prompt = len(tokenizer(build_prompt(tokenizer, "hi")).input_ids)
        ids = list(range(n_prompt)) + tokenizer.encode("done") + [tokenizer.eos_token_id]
        _, result = self._run(tokenizer, ids)
        assert "<|im_end|>" not in result.text

    def test_counts_are_honest(self, tokenizer):
        n_prompt = len(tokenizer(build_prompt(tokenizer, "hi")).input_ids)
        completion = tokenizer.encode(" a b c")
        _, result = self._run(tokenizer, list(range(n_prompt)) + completion)
        assert result.prompt_tokens == n_prompt
        assert result.completion_tokens == len(completion)

    def test_finish_reason_stop_on_eos(self, tokenizer):
        n_prompt = len(tokenizer(build_prompt(tokenizer, "hi")).input_ids)
        ids = list(range(n_prompt)) + tokenizer.encode("hey") + [tokenizer.eos_token_id]
        _, result = self._run(tokenizer, ids, max_new_tokens=64)
        assert result.finish_reason == "stop"

    def test_finish_reason_length_without_eos(self, tokenizer):
        n_prompt = len(tokenizer(build_prompt(tokenizer, "hi")).input_ids)
        _, result = self._run(tokenizer, list(range(n_prompt)) + [7, 8, 9], max_new_tokens=3)
        assert result.finish_reason == "length"

    def test_sampling_kwargs_reach_generate(self, tokenizer):
        model = StubModel([1, 2, 3])
        complete(model, tokenizer, "hi", 5, temperature=0.7, top_k=20)
        assert model.generate_kwargs["do_sample"] is True
        assert model.generate_kwargs["temperature"] == 0.7
        assert model.generate_kwargs["top_k"] == 20
        assert model.generate_kwargs["max_new_tokens"] == 5

    def test_attention_mask_is_passed(self, tokenizer):
        # Without it generate() falls back to "everything is real", which is wrong
        # the moment anything is padded -- and it warns on every call.
        model = StubModel([1, 2, 3])
        complete(model, tokenizer, "hi", 5, temperature=0.0)
        assert "attention_mask" in model.generate_kwargs

    def test_runs_against_a_real_qwen3(self, tokenizer, tiny_model):
        # Random weights, so the text is gibberish; what matters is that the whole
        # path executes and the counts stay self-consistent.
        result = complete(tiny_model, tokenizer, "hello", 6, temperature=0.0)
        assert isinstance(result.text, str)
        assert result.completion_tokens <= 6
        assert result.prompt_tokens > 0
        assert result.finish_reason in {"stop", "length"}


class TestCompleteStream:
    def test_yields_strings(self, tokenizer, tiny_model):
        pieces = list(complete_stream(tiny_model, tokenizer, "hello", 6, temperature=0.0))
        assert all(isinstance(p, str) for p in pieces)

    def test_no_empty_chunks(self, tokenizer, tiny_model):
        # TextIteratorStreamer emits a couple of "" before the first real token.
        # An empty SSE chunk is pure noise on the wire.
        pieces = list(complete_stream(tiny_model, tokenizer, "hello", 6, temperature=0.0))
        assert all(p != "" for p in pieces)

    def test_no_special_tokens(self, tokenizer, tiny_model):
        text = "".join(complete_stream(tiny_model, tokenizer, "hello", 8, temperature=0.0))
        assert "<|im_end|>" not in text and "<|endoftext|>" not in text

    def test_is_lazy_not_precomputed(self, tokenizer, tiny_model):
        # Must be a generator: main.py pulls one piece at a time so the event loop
        # stays free between decode steps. Returning a list would block the whole
        # generation before the first byte reaches the client.
        import inspect

        gen = complete_stream(tiny_model, tokenizer, "hello", 6, temperature=0.0)
        assert inspect.isgenerator(gen)
        gen.close()

    def test_matches_complete_under_greedy_decoding(self, tokenizer, tiny_model):
        # Same inputs, no sampling -> streaming and one-shot must agree. If they
        # differ, one of the two is slicing or decoding wrong.
        streamed = "".join(
            complete_stream(tiny_model, tokenizer, "hello", 8, temperature=0.0)
        )
        oneshot = complete(tiny_model, tokenizer, "hello", 8, temperature=0.0)
        assert streamed == oneshot.text

    def test_thread_does_not_leak(self, tokenizer, tiny_model):
        import threading

        before = threading.active_count()
        list(complete_stream(tiny_model, tokenizer, "hello", 6, temperature=0.0))
        assert threading.active_count() == before
