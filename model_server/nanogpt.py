"""nano-GPT: a from-scratch decoder-only transformer, small enough to train on a laptop CPU.

Milestone 2 skeleton (see docs/nanogpt-tutorial.md). The module wiring (__init__,
checkpoint I/O, generation loop) is given so you can focus on the parts worth
learning: the forward passes and the sampling step. Everything marked TODO raises
NotImplementedError; `pytest test_nanogpt.py` is the grader -- implement until it
is green. Reference implementations live in the tutorial, section by section.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CharTokenizer:
    """Character-level tokenizer: one char = one token.

    No BPE, no external vocab file -- the vocabulary is just the sorted set of
    characters seen in the training corpus. This keeps the whole tokenizer at
    ~10 lines, at the cost of longer sequences than a real LLM tokenizer.
    """

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        # map each char to its id. Chars not in the vocab (the API
        # accepts arbitrary prompts!) must be silently dropped, not crash.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        # map ids back to chars and join.
        return "".join([self.itos[i] for i in ids])


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # max context length (in tokens = chars)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        # One fused projection producing Q, K, V for all heads at once.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask, precomputed once. Shape (1, 1, T, T) so it
        # broadcasts over (batch, head). persistent=False keeps it out of checkpoints.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3C (Q, K, V)
        hs = C // self.n_head
        q, k, v = qkv.split(C, dim=2)   # Split according to the last dimension C
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # # (B, nh, T, hs)

        attn = q @ k.transpose(-2, -1) / math.sqrt(hs)  # (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)
        attn = F.softmax(attn, dim=-1)   # (B, nh, T, T)
        attn = self.attn_dropout(attn)  # (B, nh, T, T)

        y = attn @ v    # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, nh, hs) -> (B, T, C)
        return self.resid_dropout(self.c_proj(y))



class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: LayerNorm goes *before* the sublayer, and the
    residual path around it stays clean -- this is what lets gradients flow through
    deep stacks without warmup tricks (GPT-2 style, vs the original post-norm)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residuals: norm goes inside each branch, the residual path stays clean.
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class NanoGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: input embedding and output projection share one matrix.
        # Saves ~10% of params here and is what GPT-2/Qwen actually do.
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids. Returns (logits (B, T, vocab_size), loss or None)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.wte(idx)  # (B, T, C)
        pos_emb = self.wpe(pos)  # (T, C), broadcasts over batch
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = F.cross_entropy(torch.flatten(logits, end_dim=-2), targets.view(-1)) if targets is not None else None

        return logits, loss


    @torch.no_grad()
    def _next_token(
        self, idx: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Sample ONE next token given the context. idx: (B, T) -> returns (B, 1)."""
        # Crop to the last block_size tokens -- the position embeddings don't reach further.
        idx_cond = idx[:, -self.cfg.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            logits = F.softmax(logits, dim=-1)
            return torch.multinomial(logits, 1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive loop: sample, append, repeat. Returns prompt + completion."""
        for _ in range(max_new_tokens):
            nxt = self._next_token(idx, temperature, top_k)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Same loop, but yields each token id as soon as it is sampled (B must be 1).
        This is what the SSE endpoint iterates -- streaming falls out of the
        autoregressive loop for free, one yield per decode step."""
        for _ in range(max_new_tokens):
            nxt = self._next_token(idx, temperature, top_k)
            idx = torch.cat((idx, nxt), dim=1)
            yield int(nxt.item())


# ---------------------------------------------------------------------------
# Plumbing used by train.py and main.py -- given, nothing to implement below.
# ---------------------------------------------------------------------------


def save_checkpoint(model: NanoGPT, tokenizer: CharTokenizer, path: str) -> None:
    # Config and vocab travel inside the checkpoint so the server needs exactly
    # one file (and one env var) to reconstruct everything.
    torch.save(
        {"config": asdict(model.cfg), "model": model.state_dict(), "chars": tokenizer.chars},
        path,
    )


def load_model(path: str) -> tuple[NanoGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu")
    model = NanoGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()  # switches dropout off -- forgetting this makes sampling noticeably worse
    return model, CharTokenizer(ckpt["chars"])


def _prompt_ids(tokenizer: CharTokenizer, prompt: str) -> list[int]:
    # Arbitrary API prompts may contain zero known chars; the model still needs
    # at least one context token, so fall back to something in-vocab.
    return tokenizer.encode(prompt) or tokenizer.encode("\n") or [0]


def complete(
    model: NanoGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """Blocking one-shot completion. main.py calls this via asyncio.to_thread so
    the CPU-bound forward passes never stall the event loop."""
    ids = _prompt_ids(tokenizer, prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0, len(ids):].tolist())


def complete_stream(
    model: NanoGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """Blocking generator yielding decoded text piece by piece (one char/token each)."""
    ids = _prompt_ids(tokenizer, prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    for token_id in model.generate_stream(idx, max_new_tokens, temperature, top_k):
        yield tokenizer.decode([token_id])
