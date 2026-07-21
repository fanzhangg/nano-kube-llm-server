"""Train nano-GPT on tiny-shakespeare, character-level. CPU-friendly: the default
config (~0.8M params) reaches val loss ~1.7 in a few minutes on a laptop.

Usage:
    python train.py                     # downloads data on first run, trains, saves
    python train.py --iters 500         # quick smoke run
    python train.py --sample-only       # load existing checkpoint and print a sample

The core loop is get_batch() (random windows, target = input shifted by one) and
train_step() (forward -> zero_grad -> backward -> step); the harness around them
handles data download, eval loop, checkpointing, and sampling.
"""

import argparse
import os
import time
import urllib.request

import torch

from nanogpt import CharTokenizer, GPTConfig, NanoGPT, load_model, save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(HERE, "data", "tinyshakespeare.txt")
CKPT_PATH = os.path.join(HERE, "checkpoints", "nanogpt.pt")


def load_corpus() -> str:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences from the corpus.

    data is the whole corpus as a 1-D tensor of token ids. Language modeling
    targets are just the input shifted by one: for a slice data[i : i+block_size],
    the target is data[i+1 : i+block_size+1] -- the model learns "given everything
    so far, predict the NEXT char" at every position simultaneously.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))   # random start offsets
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])  # target = input shifted by 1
    return x, y

def train_step(
    model: NanoGPT, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> float:
    """One optimization step. Returns the loss as a float."""
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)   # clear the gradient of last step
    loss.backward() # back propagation
    optimizer.step()    # update the parameter with gradient descent
    return loss.item()


@torch.no_grad()
def estimate_loss(
    model: NanoGPT, data: torch.Tensor, block_size: int, batch_size: int, iters: int = 40
) -> float:
    """Average loss over several random batches -- one batch is too noisy to read."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-only", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    if args.sample_only:
        model, tokenizer = load_model(CKPT_PATH)
        _print_sample(model, tokenizer)
        return

    text = load_corpus()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # 90/10 train/val split. Val loss is what tells you the model is learning
    # language rather than memorizing -- watch the gap between the two.
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(data)} chars, vocab_size={tokenizer.vocab_size}")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = NanoGPT(cfg)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    started = time.time()
    for step in range(1, args.iters + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size)
        loss = train_step(model, optimizer, x, y)
        if step % args.eval_interval == 0 or step == args.iters:
            val = estimate_loss(model, val_data, args.block_size, args.batch_size)
            print(
                f"step {step:5d}  train_loss {loss:.3f}  val_loss {val:.3f}  "
                f"({time.time() - started:.0f}s)"
            )

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    save_checkpoint(model, tokenizer, CKPT_PATH)
    print(f"saved {CKPT_PATH}")
    model.eval()
    _print_sample(model, tokenizer)


def _print_sample(model: NanoGPT, tokenizer: CharTokenizer) -> None:
    from nanogpt import complete

    print("--- sample ---")
    print(complete(model, tokenizer, "\n", max_new_tokens=400, temperature=0.8, top_k=40))


if __name__ == "__main__":
    main()
