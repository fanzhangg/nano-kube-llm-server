"""Grader for the Milestone 2 training loop (train.py).

Covers the two functions the skeleton asks you to implement -- get_batch()
(random windows, target = input shifted by one) and train_step() (forward ->
zero_grad -> backward -> step) -- plus the estimate_loss() eval helper. These
run on synthetic tensors, so they never touch load_corpus() / the network.

Work through train.py until `pytest test_train.py -v` is green.
"""

import pytest

torch = pytest.importorskip("torch", reason="milestone 2 needs torch (CPU wheel is fine)")

from nanogpt import GPTConfig, NanoGPT  # noqa: E402
from train import estimate_loss, get_batch, train_step  # noqa: E402

VOCAB = 11


def tiny_model(vocab_size: int = VOCAB) -> NanoGPT:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    return NanoGPT(cfg)


class TestGetBatch:
    def test_shapes_and_dtype(self):
        data = torch.arange(100)
        x, y = get_batch(data, block_size=8, batch_size=4)
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert x.dtype == data.dtype  # ids must stay integer, not cast to float

    def test_target_is_input_shifted_by_one(self):
        # THE language-modeling contract: y[b, t] is the token that follows
        # x[b, t]. Using a contiguous 0..N-1 corpus makes the shift exact --
        # every sampled window is a run of consecutive ints, so y == x + 1.
        # Fails on an off-by-one (y == x, or shifted the wrong direction).
        data = torch.arange(200)
        x, y = get_batch(data, block_size=8, batch_size=16)
        assert torch.equal(y, x + 1)
        assert torch.equal(x[:, 1:], y[:, :-1])  # windows overlap by one, as they must

    def test_never_indexes_out_of_bounds(self):
        # High = len(data) - block_size - 1 is the subtle bit: the target window
        # reaches i + block_size, so a too-large start silently reads past the
        # end. Small corpus + many draws makes randint hit the top offset.
        data = torch.arange(20)
        for _ in range(200):
            x, y = get_batch(data, block_size=8, batch_size=8)
            assert x.min() >= 0 and x.max() < len(data)
            assert y.min() >= 0 and y.max() < len(data)

    def test_batches_are_random(self):
        # Two draws from a large corpus should almost never coincide; an identical
        # pair means the start offsets aren't actually being sampled.
        data = torch.arange(1000)
        torch.manual_seed(1)
        a, _ = get_batch(data, block_size=8, batch_size=8)
        b, _ = get_batch(data, block_size=8, batch_size=8)
        assert not torch.equal(a, b)


class TestTrainStep:
    def _batch(self, seed: int = 1):
        torch.manual_seed(seed)
        x = torch.randint(0, VOCAB, (4, 15))
        y = torch.randint(0, VOCAB, (4, 15))
        return x, y

    def test_returns_python_float(self):
        model = tiny_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, y = self._batch()
        loss = train_step(model, optimizer, x, y)
        assert isinstance(loss, float)  # loss.item(), not the 0-d tensor

    def test_updates_parameters(self):
        # One step must actually move the weights: forward, backward, and
        # optimizer.step() all wired up. Snapshot a param before, compare after.
        model = tiny_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        before = next(model.parameters()).clone()
        x, y = self._batch()
        train_step(model, optimizer, x, y)
        after = next(model.parameters())
        assert not torch.allclose(before, after)

    def test_repeated_steps_reduce_loss(self):
        # The whole point of the loop: repeatedly stepping on one memorizable
        # batch drives loss down. Also catches a missing zero_grad -- accumulated
        # gradients would send the loss up or NaN rather than down.
        model = tiny_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        x, y = self._batch()
        first = train_step(model, optimizer, x, y)
        for _ in range(60):
            last = train_step(model, optimizer, x, y)
        assert last < first * 0.7


class TestEstimateLoss:
    def test_returns_float_and_restores_train_mode(self):
        # estimate_loss runs under no_grad in eval mode but must leave the model
        # back in train() so the loop keeps learning after an eval interval.
        model = tiny_model()
        model.train()
        data = torch.randint(0, VOCAB, (500,))
        loss = estimate_loss(model, data, block_size=8, batch_size=4, iters=5)
        assert isinstance(loss, float)
        assert loss > 0
        assert model.training  # eval() then train() -- must end in train mode

    def test_does_not_track_gradients(self):
        # The @torch.no_grad decorator means no param should come out with a grad.
        model = tiny_model()
        model.train()
        model.zero_grad(set_to_none=True)
        data = torch.randint(0, VOCAB, (500,))
        estimate_loss(model, data, block_size=8, batch_size=4, iters=5)
        assert all(p.grad is None for p in model.parameters())
