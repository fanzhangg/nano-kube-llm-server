"""Qwen3 weight loading. Prompting and execution live in runners.QwenRunner."""

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
