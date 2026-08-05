"""Model-runner contract: HOW a batch becomes tokens.

Pairs with test_batching.py (WHAT runs). Everything model-specific is graded
here, which is the point of the split -- adding a model means adding a runner and
a section to this file, and touching nothing else.

QwenRunner is graded against a stub that records the tensors it was handed: what
matters is padding side, position ids, and the prefill/decode input shapes, none
of which need real weights. MockRunner is graded on the contract test_main.py has
asserted since Milestone 1.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="QwenRunner needs torch")

from batching import Sequence  # noqa: E402
from runners import ForwardBatch, MockRunner, QwenRunner  # noqa: E402

VOCAB = 16
EOS, ENDOFTEXT = 151645, 151643


def seq(prompt_len: int, **kw) -> Sequence:
    return Sequence(
        prompt_ids=list(range(1, prompt_len + 1)),
        max_new_tokens=kw.pop("max_new_tokens", 4),
        temperature=kw.pop("temperature", 0.0),
        top_k=kw.pop("top_k", None),
        **kw,
    )


class StubModel:
    """Records every call; returns logits whose argmax we control."""

    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=[EOS, ENDOFTEXT])

    def __init__(self, forced=None):
        self.forced = forced or (lambda call, row: 7)
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        index = len(self.calls) - 1
        batch, width = kw["input_ids"].shape
        logits = torch.zeros(batch, width, VOCAB)
        for row in range(batch):
            logits[row, -1, self.forced(index, row)] = 100.0
        return SimpleNamespace(
            logits=logits, past_key_values=SimpleNamespace(tag=f"cache-{index}")
        )

    @property
    def last(self):
        return self.calls[-1]


class StubTokenizer:
    eos_token_id = EOS
    pad_token_id = ENDOFTEXT

    def apply_chat_template(self, messages, **kw):
        return f"<im>{messages[0]['content']}</im>"

    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=[ord(c) % 50 + 1 for c in text])

    def decode(self, ids, skip_special_tokens=False):
        keep = [i for i in ids if not (skip_special_tokens and i in {EOS, ENDOFTEXT})]
        return "".join(chr(ord("a") + i % 26) for i in keep)


@pytest.fixture
def runner():
    return QwenRunner(StubModel(), StubTokenizer())


# ---------------------------------------------------------------------------
# QwenRunner -- stop tokens
# ---------------------------------------------------------------------------


def test_resolves_every_stop_token_not_just_the_advertised_one(runner):
    """Qwen3 has two. tokenizer.eos_token_id only reports <|im_end|>; missing
    <|endoftext|> makes a natural stop run on to max_new_tokens and report
    "length", which quietly undoes an honest finish_reason."""
    assert runner.eos_token_ids == {EOS, ENDOFTEXT}


def test_falls_back_to_the_tokenizer_when_there_is_no_generation_config():
    model = StubModel()
    model.generation_config = None

    assert QwenRunner(model, StubTokenizer()).eos_token_ids == {EOS}


# ---------------------------------------------------------------------------
# QwenRunner -- padding
# ---------------------------------------------------------------------------


def test_pads_on_the_left_not_the_right(runner):
    """Decode reads logits at the last column; right padding would sample the
    continuation of <pad> for every short row."""
    input_ids, _, _ = runner._pad_and_stack([seq(3), seq(5)])

    assert input_ids.shape == (2, 5)
    assert input_ids[0].tolist() == [ENDOFTEXT, ENDOFTEXT, 1, 2, 3]
    assert input_ids[1].tolist() == [1, 2, 3, 4, 5]


def test_attention_mask_is_zero_exactly_on_pads(runner):
    _, mask, _ = runner._pad_and_stack([seq(3), seq(5)])

    assert mask[0].tolist() == [0, 0, 1, 1, 1]
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


def test_position_ids_skip_the_padding(runner):
    """RoPE reads positions from this tensor. A row padded by two must still start
    at position 0 -- getting this wrong crashes nothing and is invisible at
    batch=1, it only degrades output once ragged sequences share a batch."""
    _, _, positions = runner._pad_and_stack([seq(3), seq(5)])

    assert positions[0].tolist() == [0, 0, 0, 1, 2]
    assert positions[1].tolist() == [0, 1, 2, 3, 4]


def test_padding_covers_generated_tokens_too(runner):
    running = seq(2)
    running.output_ids = [9, 9]

    input_ids, _, _ = runner._pad_and_stack([running])

    assert input_ids.tolist() == [[1, 2, 9, 9]]


def test_returns_int64_tensors(runner):
    """Embedding lookups and RoPE both require integer indices."""
    assert all(t.dtype == torch.int64 for t in runner._pad_and_stack([seq(3), seq(5)]))


# ---------------------------------------------------------------------------
# QwenRunner -- sampling
# ---------------------------------------------------------------------------


def test_greedy_when_temperature_is_zero(runner):
    logits = torch.zeros(1, VOCAB)
    logits[0, 11] = 5.0

    assert runner._sample(logits, [seq(2, temperature=0.0)]) == [11]


def test_each_row_uses_its_own_sampling_params(runner):
    """Rows are different HTTP requests that happened to share a forward pass."""
    logits = torch.zeros(2, VOCAB)
    logits[0, 3] = 10.0
    logits[1, 12] = 10.0

    picked = runner._sample(
        logits, [seq(2, temperature=0.0), seq(2, temperature=0.9, top_k=1)]
    )

    assert picked == [3, 12]


def test_top_k_restricts_the_candidate_set(runner):
    logits = torch.zeros(1, VOCAB)
    logits[0, 5] = 1.0
    logits[0, 6] = 0.9

    for _ in range(20):
        assert runner._sample(logits, [seq(2, temperature=1.0, top_k=1)]) == [5]


def test_unset_top_k_does_not_filter_everything(runner):
    """None / 0 / negative all mean "unset"; 0 must not read as "zero candidates"."""
    logits = torch.zeros(1, VOCAB)
    logits[0, 4] = 50.0

    for unset in (None, 0, -1):
        assert runner._sample(logits, [seq(2, temperature=0.5, top_k=unset)]) == [4]


def test_sampling_returns_plain_ints(runner):
    assert all(isinstance(t, int) for t in runner._sample(torch.randn(3, VOCAB), [seq(2)] * 3))


# ---------------------------------------------------------------------------
# QwenRunner -- execute
# ---------------------------------------------------------------------------


def test_prefill_feeds_the_whole_padded_batch(runner):
    runner.execute(ForwardBatch(seqs=[seq(3), seq(5)], is_prefill=True))

    assert runner.model.last["input_ids"].shape == (2, 5)
    assert runner.model.last["past_key_values"] is None
    assert runner.model.last["use_cache"] is True


def test_decode_feeds_only_the_newest_column(runner):
    running = seq(3)
    runner.execute(ForwardBatch(seqs=[running], is_prefill=True))
    running.output_ids = [9]

    runner.execute(ForwardBatch(seqs=[running], is_prefill=False))

    assert runner.model.last["input_ids"].tolist() == [[9]]
    assert runner.model.last["position_ids"].tolist() == [[3]]


def test_decode_passes_the_attention_mask_in_full(runner):
    """The mask says what may be ATTENDED TO, not what is being fed. Slicing it
    alongside input_ids makes every sequence attend only to its own newest token:
    the model forgets its context and emits fluent nonsense, silently."""
    running = seq(3)
    runner.execute(ForwardBatch(seqs=[running], is_prefill=True))
    running.output_ids = [9]

    runner.execute(ForwardBatch(seqs=[running], is_prefill=False))

    assert runner.model.last["attention_mask"].shape == (1, 4)


def test_decode_reuses_the_cache_from_the_previous_tick(runner):
    running = seq(3)
    runner.execute(ForwardBatch(seqs=[running], is_prefill=True))
    running.output_ids = [9]

    runner.execute(ForwardBatch(seqs=[running], is_prefill=False))

    assert runner.model.last["past_key_values"].tag == "cache-0"


def test_prefill_discards_the_cache_it_was_holding(runner):
    """is_prefill is the loop telling the runner its rows changed."""
    running = seq(3)
    runner.execute(ForwardBatch(seqs=[running], is_prefill=True))

    runner.execute(ForwardBatch(seqs=[running, seq(4)], is_prefill=True))

    assert runner.model.last["past_key_values"] is None


def test_reset_drops_the_cache(runner):
    runner.execute(ForwardBatch(seqs=[seq(3)], is_prefill=True))
    runner.reset()

    runner.execute(ForwardBatch(seqs=[seq(3)], is_prefill=False))

    assert runner.model.last["past_key_values"] is None


def test_execute_returns_one_token_per_row(runner):
    picked = runner.execute(ForwardBatch(seqs=[seq(3), seq(5)], is_prefill=True))

    assert picked == [7, 7]


def test_encode_applies_the_chat_template(runner):
    """A bare string makes Qwen3 continue the user's turn instead of answering."""
    assert runner.encode("hi") == runner.tokenizer("<im>hi</im>").input_ids


def test_decode_strips_special_tokens(runner):
    assert runner.decode([5, EOS]) == runner.tokenizer.decode([5])


# ---------------------------------------------------------------------------
# MockRunner -- no torch, same scheduler
# ---------------------------------------------------------------------------


def test_mock_counts_prompt_tokens_as_characters():
    """test_main.py has asserted prompt_tokens == len(prompt) since Milestone 1."""
    assert len(MockRunner().encode("hello")) == 5


def test_mock_never_stops_early():
    """No EOS, so every request finishes on max_new_tokens and reports "length" --
    exactly the Milestone 1 mock's contract."""
    assert MockRunner().eos_token_ids == set()


def test_mock_emits_one_token_per_row():
    mock = MockRunner()

    assert mock.execute(ForwardBatch(seqs=[seq(2), seq(2)], is_prefill=True)) == [0, 0]


def test_mock_decodes_incrementally_so_streaming_works():
    """The engine yields decode(output)[sent:]. If the mock's text were not a
    growing prefix, streamed reassembly would not match the one-shot text."""
    mock = MockRunner()
    texts = [mock.decode([0] * n) for n in range(4)]

    for shorter, longer in zip(texts, texts[1:]):
        assert longer.startswith(shorter)


def test_the_mock_path_imports_no_torch():
    """The ~150MB mock image has no torch in it at all, yet runs the real
    Scheduler -- which is why the Kubernetes demo measures the genuine admission
    rule instead of a semaphore pretending to be one.

    Checked structurally rather than by text: torch must not appear at module
    scope in either file on the mock's import path. QwenRunner may import it
    inside its methods, and does.
    """
    import ast
    import pathlib

    for filename in ("batching.py", "runners.py"):
        tree = ast.parse(pathlib.Path(filename).read_text())
        for node in tree.body:  # module scope only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "torch" for n in names), (
                f"{filename} imports torch at module scope"
            )


# ---------------------------------------------------------------------------
# QwenRunner against a REAL transformers model
# ---------------------------------------------------------------------------
#
# The stubs above grade the contract; they cannot grade whether transformers
# actually accepts these tensors. A ~5M-param randomly-initialised Qwen3 has the
# same architecture and the same forward signature as the 0.6B, runs on CPU in
# seconds, and needs no download. The weights are noise, so the OUTPUT is
# meaningless -- what is under test is that the plumbing is real.


@pytest.fixture(scope="module")
def tiny_runner():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, eos_token_id=255, pad_token_id=0,
    )
    model = Qwen3ForCausalLM(config).eval()
    runner = QwenRunner(model, StubTokenizer())
    # StubTokenizer carries the REAL Qwen pad id (151643), which is outside this
    # 256-token vocab and would blow up the embedding lookup.
    runner.pad_id = 0
    return runner


def test_real_model_accepts_a_ragged_prefill(tiny_runner):
    picked = tiny_runner.execute(ForwardBatch(seqs=[seq(3), seq(7)], is_prefill=True))

    assert len(picked) == 2
    assert all(0 <= t < 256 for t in picked)


def test_real_model_accepts_a_decode_step_against_the_live_cache(tiny_runner):
    a, b = seq(3), seq(7)
    first = tiny_runner.execute(ForwardBatch(seqs=[a, b], is_prefill=True))
    a.output_ids, b.output_ids = [first[0]], [first[1]]

    second = tiny_runner.execute(ForwardBatch(seqs=[a, b], is_prefill=False))

    assert len(second) == 2


def test_batched_greedy_agrees_with_unbatched_greedy(tiny_runner):
    """A short sequence must generate the same tokens alone as it does beside a
    longer one, where it is left-padded.

    HONEST LIMIT, verified by mutation: with RANDOM weights this does NOT catch a
    position_ids bug. Sabotaging position_ids to a naive arange (22 pads, 6 steps)
    leaves this assertion green -- untrained logits are arbitrary enough that RoPE
    perturbations do not move the argmax. `test_position_ids_skip_the_padding`
    above is what actually guards that, and real weights on Day 13 are what would
    show the degradation end to end.

    What this DOES buy: transformers really accepts these tensors -- shapes,
    dtypes, mask width, cache hand-off -- which no stub can confirm.
    """
    def greedy(seqs, steps):
        for s in seqs:
            s.output_ids = []
        out = [[] for _ in seqs]
        for i in range(steps):
            picked = tiny_runner.execute(ForwardBatch(seqs=seqs, is_prefill=(i == 0)))
            for row, token in enumerate(picked):
                seqs[row].output_ids.append(token)
                out[row].append(token)
        return out

    alone = greedy([seq(2)], steps=6)[0]
    batched = greedy([seq(2), seq(24)], steps=6)[0]

    assert batched == alone, "padding / position_ids / mask disagree once ragged"
