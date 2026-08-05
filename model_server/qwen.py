"""Qwen3 loader and chat template. Execution lives in runners.QwenRunner.

This module owns "how to generate" and nothing else. It deliberately imports no
FastAPI: main.py's Engine ABC adapts these functions to HTTP, which is what makes
the engine swappable without touching a single request handler.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3-0.6B: 28 layers, 16 query heads / 8 KV heads (GQA), 40960 max context.
# ~1.2GB in bfloat16.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def pick_dtype(device: str) -> torch.dtype:
    """Weight dtype for a device string like "cuda" / "cuda:0" / "cpu".

    bfloat16 over float16 on GPU: same width, but bf16 keeps float32's exponent
    range and drops mantissa bits instead, so it needs no loss scaling and will
    not silently produce inf/NaN. float32 on CPU because most CPU kernels fall
    back to a slow emulated path for bf16.
    """
    if device.startswith("cuda"):
        return torch.bfloat16
    else:
        return torch.float32


def build_prompt(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    """Wrap a raw prompt in the control tokens Qwen3 was fine-tuned on.

    Qwen3 is instruct tuned and expects `<|im_start|>user ... <|im_end|>
    <|im_start|>assistant`. A bare string produces a model that continues the
    user's turn instead of answering -- which reads like a broken model but is a
    broken prompt.

    enable_thinking=False does not remove the reasoning block; it prefills an
    empty one (`<think>\\n\\n</think>`) so the model treats reasoning as already
    finished and goes straight to the answer. Left enabled, Qwen3 will reason for
    hundreds of tokens first.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_qwen(model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
    """Load tokenizer + weights. Returns (model, tokenizer).

    `model_id` is a HuggingFace repo id, not a local path: the first call
    downloads ~1.2GB into ~/.cache/huggingface.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=pick_dtype(device),  # transformers v5 renamed torch_dtype -> dtype
        device_map=device,
    )
    model.eval()
    return model, tokenizer
