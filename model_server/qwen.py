"""Inference engine for Qwen3 checkpoints served through the HuggingFace stack.

This module owns "how to generate" and nothing else. It deliberately imports no
FastAPI: main.py's Engine ABC adapts these functions to HTTP, which is what makes
the engine swappable without touching a single request handler.
"""

import threading
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Qwen3-0.6B: 28 layers, 16 query heads / 8 KV heads (GQA), 40960 max context.
# ~1.2GB in bfloat16.
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class Generation:
    """A completed generation, with the token counts the OpenAI usage block needs."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" -> model emitted EOS | "length" -> hit max_new_tokens


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


def sampling_kwargs(temperature: float, top_k: int | None) -> dict:
    """Translate OpenAI-style sampling params into `model.generate` kwargs.

    The two schemas disagree about how to express "no randomness": OpenAI says
    temperature=0, while generate() wants do_sample=False and rejects
    temperature=0 outright. Greedy mode returns do_sample alone -- passing
    sampling knobs that cannot take effect makes generate() warn on every call.

    top_k is an optional vLLM extension rather than part of the OpenAI schema, so
    None/0/negative all mean "unset" (0 would otherwise read as a filter down to
    zero candidate tokens).
    """
    if temperature <= 0:
        return {"do_sample": False}

    kwargs = {"do_sample": True, "temperature": temperature}
    if top_k and top_k > 0:
        kwargs["top_k"] = top_k
    return kwargs


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


def complete(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int | None = 20,
    enable_thinking: bool = False,
) -> Generation:
    """Blocking one-shot completion. Call via asyncio.to_thread from async code."""
    prompt_text = build_prompt(tokenizer, prompt, enable_thinking)
    encoding = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **encoding,
        max_new_tokens=max_new_tokens,
        **sampling_kwargs(temperature, top_k),
    )

    # generate() returns prompt + completion concatenated; slicing off the prompt
    # is what keeps the templated control tokens out of the API response.
    prompt_tokens = encoding.input_ids.shape[1]
    generated_ids = output[0, prompt_tokens:]

    return Generation(
        text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        prompt_tokens=prompt_tokens,
        completion_tokens=len(generated_ids),
        finish_reason=(
            "stop" if (generated_ids == tokenizer.eos_token_id).any() else "length"
        ),
    )


def complete_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int | None = 20,
    enable_thinking: bool = False,
):
    """Blocking generator yielding decoded text pieces as they are produced.

    generate() is push-based (it calls streamer.put per step) while callers need
    to pull, so TextIteratorStreamer bridges the two as a blocking queue with an
    iterator face. It has to run on a separate thread: generate() does not return
    until generation finishes, so put() would otherwise block against a queue
    nobody is draining.
    """
    prompt_text = build_prompt(tokenizer, prompt, enable_thinking)
    encoding = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    # skip_prompt=True replaces the manual prompt slice done in complete().
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **encoding,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            **sampling_kwargs(temperature, top_k),
        ),
    )
    thread.start()

    try:
        for piece in streamer:
            # The streamer emits a couple of "" before the first real token,
            # and an empty SSE chunk is noise on the wire.
            if piece:
                yield piece
    finally:
        # Runs on client disconnect too, when the generator is closed early.
        thread.join()


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
