"""Model runners: everything model-specific lives below this line.

The scheduler decides WHAT runs; a runner decides HOW. One struct crosses the
boundary. This is the split vLLM and SGLang both arrive at:

    here                  vLLM                        SGLang
    ------------------    ------------------------    -----------------------
    Scheduler             Scheduler                   Scheduler
    ForwardBatch          SchedulerOutput             ForwardBatch
    is_prefill            (chunked prefill)           ForwardMode.EXTEND/DECODE
    ModelRunner.execute   GPUModelRunner.execute_m.   ModelRunner.forward
    BatchingEngine        AsyncLLM frontend           TokenizerManager

Adding a model means writing one ModelRunner subclass. main.py, engines.py and
batching.py do not change, and never learn the model's name.

torch is imported per-runner, not at module scope: MockRunner has to work in the
~150MB image that has no torch in it at all.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from batching import Sequence


@dataclass
class ForwardBatch:
    """One tick's work, in the shape a runner needs.

    `is_prefill` is decided by run_loop (membership changed -> the KV cache no
    longer describes these rows) and handed down, rather than inferred inside the
    runner from "is my cache None". Keeping it explicit is what makes a future
    mixed batch -- some rows prefilling, some decoding -- expressible at all.
    """

    seqs: list[Sequence]
    is_prefill: bool


class ModelRunner(ABC):
    """Owns tokenization, execution and sampling for exactly one model."""

    #: Every token id that means "stop" for this model. Empty = no EOS at all.
    eos_token_ids: set[int] = set()

    @abstractmethod
    def encode(self, prompt: str) -> list[int]:
        """Prompt text -> token ids, including any chat template."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Token ids -> text, with special tokens stripped."""

    @abstractmethod
    def execute(self, batch: ForwardBatch) -> list[int]:
        """Advance every sequence by one token. Returns one token id per row."""

    def reset(self) -> None:
        """Drop any cached execution state. Called when the loop goes idle."""


# ---------------------------------------------------------------------------
# Qwen3
# ---------------------------------------------------------------------------


class QwenRunner(ModelRunner):
    """Qwen3 through HuggingFace weights, driven forward pass by forward pass.

    No generate(): its loop cannot be interrupted to add or drop a sequence,
    which is the one thing continuous batching must do. So we own the KV cache.
    """

    def __init__(self, model, tokenizer, enable_thinking: bool = False):
        self.model, self.tokenizer = model, tokenizer
        self.enable_thinking = enable_thinking
        self.pad_id = tokenizer.pad_token_id or 0
        self.eos_token_ids = self._resolve_eos_ids(model, tokenizer)
        self._cache = None

    @staticmethod
    def _resolve_eos_ids(model, tokenizer) -> set[int]:
        """Every stop token, not just the one the tokenizer advertises.

        Qwen3 has TWO: <|im_end|> (151645) and <|endoftext|> (151643). Only the
        first is tokenizer.eos_token_id; the full list lives in generation_config,
        which is where generate() read it from on your behalf. Miss the second and
        a sequence that ends naturally keeps generating to max_new_tokens, trails
        degenerate text, and reports "length" -- undoing an honest finish_reason.
        """
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = tokenizer.eos_token_id
        if eos is None:
            return set()
        return set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def encode(self, prompt: str) -> list[int]:
        from qwen import build_prompt

        return self.tokenizer(build_prompt(self.tokenizer, prompt, self.enable_thinking)).input_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def reset(self) -> None:
        self._cache = None

    # -- the three pieces of one tick ---------------------------------------

    def _pad_and_stack(self, seqs: list[Sequence]):
        """Stack ragged sequences into one rectangular batch, padding on the LEFT.

        Left, not right, because decode always reads logits at the last column:
        with right padding the last column of a short row is a pad token, so you
        would sample the continuation of <pad>.

        position_ids must SKIP the pads -- RoPE positions come from this tensor,
        so a row padded by k still starts at position 0 on its first real token.
        Getting it wrong crashes nothing, is invisible at batch=1, and only
        degrades output once ragged sequences share a batch.
        """
        import torch

        device = getattr(self.model, "device", "cpu")
        width = max(len(s) for s in seqs)
        ids, masks = [], []
        for s in seqs:
            toks = s.token_ids
            pad = width - len(toks)
            ids.append([self.pad_id] * pad + toks)
            masks.append([0] * pad + [1] * len(toks))

        input_ids = torch.tensor(ids, dtype=torch.int64, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.int64, device=device)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        return input_ids, attention_mask, position_ids

    def _sample(self, logits, seqs: list[Sequence]) -> list[int]:
        """One token per row, honouring each row's OWN temperature and top_k.

        Rows are different HTTP requests that happened to share a forward pass, so
        a single vectorised softmax over the batch would apply one row's sampling
        config to all of them. A per-row Python loop costs microseconds against a
        forward pass of milliseconds.
        """
        import torch

        picked = []
        for row, s in enumerate(seqs):
            row_logits = logits[row]
            if s.temperature <= 0:
                picked.append(int(torch.argmax(row_logits)))
                continue

            scaled = row_logits / s.temperature
            if s.top_k and s.top_k > 0:  # None / 0 / negative all mean "unset"
                k = min(s.top_k, scaled.size(-1))
                kept, _ = torch.topk(scaled, k)
                scaled = scaled.masked_fill(scaled < kept[-1], float("-inf"))

            probs = torch.softmax(scaled, dim=-1)
            picked.append(int(torch.multinomial(probs, 1)))

        return picked

    def execute(self, batch: ForwardBatch) -> list[int]:
        """One forward pass plus sampling. Runs on a worker thread."""
        import torch

        with torch.no_grad():
            if batch.is_prefill:
                self._cache = None

            input_ids, attention_mask, position_ids = self._pad_and_stack(batch.seqs)
            if self._cache is not None:
                # Decode: the cache holds everything but the newest token, so feed
                # only the last column. The MASK stays full-width -- it describes
                # what may be attended to, not what is being fed. Slicing it here
                # makes every sequence attend only to its own newest token: the
                # model forgets its context and emits fluent nonsense, silently.
                input_ids = input_ids[:, -1:]
                position_ids = position_ids[:, -1:]

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._cache,
                use_cache=True,
            )
            self._cache = out.past_key_values
            return self._sample(out.logits[:, -1, :], batch.seqs)


def load_qwen_runner(model_id: str, enable_thinking: bool = False) -> QwenRunner:
    from qwen import load_qwen

    model, tokenizer = load_qwen(model_id)
    return QwenRunner(model, tokenizer, enable_thinking)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

MOCK_WORD = "mock "
#: Per-tick latency, standing in for a decode step. One tick advances the WHOLE
#: batch, so eight concurrent requests now cost what one used to -- which is the
#: entire point, and is visible in the mock image without a GPU.
MOCK_TICK_SECONDS = 0.02


class MockRunner(ModelRunner):
    """No model, no torch -- but the real Scheduler and the real run_loop.

    This is why batching.py imports no torch. The mock image stays ~150MB and
    still exercises the genuine admission rule, so the Kubernetes story
    (Pending/Loading/Ready, readiness gating, queue-depth metrics, autoscaling on
    num_requests_waiting) runs the same scheduling code the GPU image runs.
    Before this, the mock faked queueing with a semaphore and the demo measured a
    simulation.

    One token per character-run of MOCK_WORD, no EOS -- so every request stops on
    max_new_tokens and reports "length", exactly as the Milestone 1 mock did.
    """

    eos_token_ids: set[int] = set()

    def encode(self, prompt: str) -> list[int]:
        # One token per character keeps prompt_tokens == len(prompt), which is the
        # contract test_main.py has asserted since Milestone 1.
        return [ord(c) for c in prompt]

    def decode(self, token_ids: list[int]) -> str:
        return MOCK_WORD * len(token_ids)

    def execute(self, batch: ForwardBatch) -> list[int]:
        # Sleeps once per TICK, not once per request: the batch advances together.
        time.sleep(MOCK_TICK_SECONDS)
        return [0] * len(batch.seqs)
