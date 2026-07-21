"""Grader for the Milestone 2 nano-GPT skeleton (docs/nanogpt-tutorial.md).

These tests define the contract for every TODO in nanogpt.py. Work through the
skeleton until `pytest test_nanogpt.py -v` is green -- each test names the
property it checks, and the causality test is the one that catches a wrong or
missing attention mask.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="milestone 2 needs torch (CPU wheel is fine)")

from nanogpt import (  # noqa: E402
    CharTokenizer,
    GPTConfig,
    NanoGPT,
    complete,
    complete_stream,
    load_model,
    save_checkpoint,
)

VOCAB = 11


def tiny_model(vocab_size: int = VOCAB, dropout: float = 0.0) -> NanoGPT:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=dropout
    )
    return NanoGPT(cfg)


class TestTokenizer:
    def test_roundtrip(self):
        tok = CharTokenizer.from_text("hello world")
        assert tok.decode(tok.encode("held")) == "held"

    def test_unknown_chars_are_dropped_not_crashed(self):
        # The API accepts arbitrary prompts; chars outside the training corpus
        # must not 500 the server.
        tok = CharTokenizer.from_text("hello world")
        assert tok.encode("h早llo!") == tok.encode("hllo")

    def test_vocab_is_sorted_and_deduplicated(self):
        tok = CharTokenizer.from_text("aba")
        assert tok.chars == ["a", "b"]
        assert tok.vocab_size == 2


class TestForward:
    def test_logits_shape(self):
        model = tiny_model()
        idx = torch.randint(0, VOCAB, (2, 10))
        logits, loss = model(idx)
        assert logits.shape == (2, 10, VOCAB)
        assert loss is None

    def test_loss_at_init_is_near_uniform(self):
        # Untrained model should be roughly uniform over the vocab: loss ~ ln(V).
        # Way above means broken init; way below means information is leaking.
        model = tiny_model()
        idx = torch.randint(0, VOCAB, (4, 12))
        targets = torch.randint(0, VOCAB, (4, 12))
        _, loss = model(idx, targets)
        assert loss is not None
        assert abs(loss.item() - math.log(VOCAB)) < 1.0

    def test_causality(self):
        # THE transformer-decoder property: changing a future token must not
        # change the logits of earlier positions. Fails if the mask is missing,
        # applied after softmax, or oriented the wrong way.
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 12))
        logits_a, _ = model(idx)
        idx_b = idx.clone()
        idx_b[0, -1] = (idx_b[0, -1] + 1) % VOCAB
        logits_b, _ = model(idx_b)
        assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
        assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)

    def test_can_overfit_one_batch(self):
        # If loss won't drop on a single memorizable batch, backprop is broken
        # somewhere even though shapes all line up.
        model = tiny_model()
        model.train()
        torch.manual_seed(1)
        x = torch.randint(0, VOCAB, (4, 15))
        y = torch.randint(0, VOCAB, (4, 15))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        _, first = model(x, y)
        for _ in range(100):
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        _, last = model(x, y)
        assert last.item() < first.item() * 0.7


class TestGenerate:
    def test_output_shape_and_range(self):
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 3))
        out = model.generate(idx, max_new_tokens=8, temperature=1.0, top_k=5)
        assert out.shape == (1, 11)
        assert torch.equal(out[:, :3], idx)  # prompt is preserved
        assert out.min() >= 0 and out.max() < VOCAB

    def test_greedy_is_deterministic(self):
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 3))
        a = model.generate(idx, max_new_tokens=8, top_k=1)
        b = model.generate(idx, max_new_tokens=8, top_k=1)
        assert torch.equal(a, b)

    def test_temperature_zero_means_greedy(self):
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 3))
        a = model.generate(idx, max_new_tokens=8, temperature=0.0)
        b = model.generate(idx, max_new_tokens=8, top_k=1)
        assert torch.equal(a, b)

    def test_stream_matches_generate(self):
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 3))
        streamed = list(model.generate_stream(idx.clone(), max_new_tokens=8, top_k=1))
        assert len(streamed) == 8
        full = model.generate(idx, max_new_tokens=8, top_k=1)
        assert streamed == full[0, 3:].tolist()

    def test_generate_beyond_block_size(self):
        # The context must be cropped to block_size inside the loop, otherwise
        # wpe indexing blows up as soon as the sequence outgrows the model.
        model = tiny_model()
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 14))  # block_size is 16
        out = model.generate(idx, max_new_tokens=10, top_k=1)
        assert out.shape == (1, 24)


class TestServerHelpers:
    def test_complete_returns_text(self):
        tok = CharTokenizer.from_text("abc \n")
        model = tiny_model(vocab_size=tok.vocab_size)
        model.eval()
        text = complete(model, tok, "ab", max_new_tokens=6, top_k=1)
        assert isinstance(text, str)
        assert len(text) == 6  # char-level: one token = one char

    def test_complete_survives_fully_unknown_prompt(self):
        tok = CharTokenizer.from_text("abc \n")
        model = tiny_model(vocab_size=tok.vocab_size)
        model.eval()
        text = complete(model, tok, "早上好", max_new_tokens=4, top_k=1)
        assert len(text) == 4

    def test_complete_stream_pieces(self):
        tok = CharTokenizer.from_text("abc \n")
        model = tiny_model(vocab_size=tok.vocab_size)
        model.eval()
        pieces = list(complete_stream(model, tok, "ab", max_new_tokens=5, top_k=1))
        assert len(pieces) == 5
        assert "".join(pieces) == complete(model, tok, "ab", max_new_tokens=5, top_k=1)


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer.from_text("abcdef \n")
        model = tiny_model(vocab_size=tok.vocab_size)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(model, tok, path)
        loaded, tok2 = load_model(path)
        assert tok2.chars == tok.chars
        assert not loaded.training  # load_model must return eval() mode
        model.eval()
        idx = torch.randint(0, tok.vocab_size, (1, 7))
        a, _ = model(idx)
        b, _ = loaded(idx)
        assert torch.allclose(a, b, atol=1e-6)
