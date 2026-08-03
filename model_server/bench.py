"""Benchmark harness for the Qwen3 engine (Milestone 3).

Not part of the skeleton -- this is a finished tool, nothing to implement here.

Separates the two phases that get conflated by a single tokens/sec number:

  prefill  -- one forward over the whole prompt. Compute bound. Sets TTFT.
  decode   -- one forward per generated token. MEMORY BANDWIDTH bound: every
              token reads all 0.6B weights out of VRAM, so the ceiling is
              (bandwidth / bytes-of-weights) tokens/sec, not FLOPs.

That distinction is the whole story on a consumer GPU, and it is why batching
is nearly free while making the model "faster" mostly is not.

Usage:
    python bench.py dtype      # fp32 vs fp16 vs bf16
    python bench.py attn       # eager vs sdpa
    python bench.py batch      # batch size 1..32 scaling
    python bench.py compile    # dynamic cache vs static cache vs CUDA graphs
    python bench.py roofline   # measured decode vs theoretical bandwidth limit
    python bench.py all
"""

import argparse
import gc
import os
import statistics
import time

# Must be set before torch is imported -- inductor reads it when its config
# module loads. The default is min(32, nproc), and worker_start_method is
# "subprocess", so each worker is a fresh interpreter importing torch (~500 MB
# RSS, no copy-on-write sharing). On a 24-core box under WSL2 (which caps the
# guest at 50% of host RAM by default) that is ~12 GB of demand and the guest
# OOM-killer takes the whole session down. 4 is plenty for a single 0.6B model.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

MODEL_ID = "Qwen/Qwen3-0.6B"
# RTX 4070 Ti: 504 GB/s spec memory bandwidth (GDDR6X, 192-bit @ 21 Gbps).
SPEC_BANDWIDTH_GBS = 504.0


def load(dtype=torch.bfloat16, attn="sdpa"):
    # device_map="cuda" streams each shard to VRAM as it is read. Loading to CPU
    # and then calling .to("cuda") materialises the whole model in host RAM
    # first, which for float32 is ~2.4 GB -- enough to OOM a memory-capped VM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation=attn, device_map="cuda"
    ).eval()
    return model


def make_batch(tokenizer, batch_size: int, prompt_tokens: int):
    """A synthetic prompt of an exact token length, so prefill cost is controlled."""
    ids = torch.full((batch_size, prompt_tokens), tokenizer.encode("the")[0], device="cuda")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@torch.no_grad()
def measure(model, enc, new_tokens: int, runs: int = 3, **gen_kwargs):
    """Returns (ttft_ms, decode_tok_s, total_tok_s) using the median run.

    min_new_tokens == max_new_tokens pins the token count so runs are comparable
    (otherwise an early EOS silently shortens a run and inflates tok/s).
    """
    base = dict(
        do_sample=False,
        min_new_tokens=new_tokens,
        max_new_tokens=new_tokens,
        pad_token_id=0,
        **gen_kwargs,
    )
    bs = enc["input_ids"].shape[0]

    # Warmup: first call allocates the cache and, when compiled, triggers the compile.
    model.generate(**enc, **{**base, "min_new_tokens": 4, "max_new_tokens": 4})
    torch.cuda.synchronize()

    ttfts, totals = [], []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, **{**base, "min_new_tokens": 1, "max_new_tokens": 1})
        torch.cuda.synchronize()
        ttfts.append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, **base)
        torch.cuda.synchronize()
        totals.append(time.perf_counter() - t0)

    ttft, total = statistics.median(ttfts), statistics.median(totals)
    # Subtract prefill so decode rate is not polluted by TTFT.
    decode_tok_s = bs * (new_tokens - 1) / (total - ttft)
    return ttft * 1000, decode_tok_s, bs * new_tokens / total


def free():
    """Reclaim VRAM after the caller has dropped its references.

    This deliberately takes no arguments. `del o` inside a helper only unbinds
    the helper's local name -- the caller's `model` still points at the model,
    so the old version of this function freed nothing and every loop below held
    two models at once. Callers must `del` their own names, then call this.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def row(label, ttft, dec, peak_gb, extra=""):
    print(f"{label:<26} {ttft:>8.1f} {dec:>12.1f} {peak_gb:>9.2f}  {extra}")


def header(title, third="decode tok/s"):
    print(f"\n=== {title} ===")
    print(f"{'config':<26} {'TTFT ms':>8} {third:>12} {'peak GB':>9}")


def bench_dtype(tokenizer, args):
    header("dtype (batch=1)")
    for name, dt in [("float32", torch.float32), ("float16", torch.float16),
                     ("bfloat16", torch.bfloat16)]:
        model = load(dt)
        enc = make_batch(tokenizer, 1, args.prompt_tokens)
        ttft, dec, _ = measure(model, enc, args.new_tokens)
        peak = torch.cuda.max_memory_allocated() / 1e9
        weights = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        row(name, ttft, dec, peak, f"weights {weights:.2f} GB")
        del model, enc
        free()


def bench_attn(tokenizer, args):
    header(f"attention impl (batch=1, prompt={args.prompt_tokens})")
    for attn in ["eager", "sdpa"]:
        model = load(torch.bfloat16, attn)
        enc = make_batch(tokenizer, 1, args.prompt_tokens)
        ttft, dec, _ = measure(model, enc, args.new_tokens)
        row(attn, ttft, dec, torch.cuda.max_memory_allocated() / 1e9)
        del model, enc
        free()

    # Attention cost is quadratic in prompt length, so the gap only opens up on
    # long prompts. At 32 tokens the two are indistinguishable.
    header("attention impl (batch=1, prompt=2048)", "decode tok/s")
    for attn in ["eager", "sdpa"]:
        model = load(torch.bfloat16, attn)
        enc = make_batch(tokenizer, 1, 2048)
        ttft, dec, _ = measure(model, enc, 32)
        row(attn, ttft, dec, torch.cuda.max_memory_allocated() / 1e9)
        del model, enc
        free()


def bench_batch(tokenizer, args):
    header("batch scaling", "total tok/s")
    model = load()
    base = None
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        try:
            enc = make_batch(tokenizer, bs, args.prompt_tokens)
            ttft, _, total = measure(model, enc, args.new_tokens)
            base = base or total
            peak = torch.cuda.max_memory_allocated() / 1e9
            row(f"batch={bs}", ttft, total, peak, f"{total / base:>5.1f}x vs bs=1")
            del enc
            free()
        except torch.cuda.OutOfMemoryError:
            print(f"batch={bs:<20} OOM")
            enc = None  # rebind rather than del: make_batch may have been what OOM'd
            free()
            break
    del model
    free()


def bench_compile(tokenizer, args):
    header("cache + compile (batch=1)")
    # 1. baseline: dynamic cache, eager python loop
    model = load()
    enc = make_batch(tokenizer, 1, args.prompt_tokens)
    ttft, dec, _ = measure(model, enc, args.new_tokens)
    row("dynamic cache", ttft, dec, torch.cuda.max_memory_allocated() / 1e9)
    del model, enc
    free()

    # 2. static cache alone: preallocated KV, no per-step reallocation. Compilation
    #    is explicitly OFF here to isolate what the cache change is worth on its
    #    own -- see the note on row 3 for why that has to be said out loud.
    model = load()
    enc = make_batch(tokenizer, 1, args.prompt_tokens)
    ttft, dec, _ = measure(model, enc, args.new_tokens,
                           cache_implementation="static", disable_compile=True)
    row("static cache", ttft, dec, torch.cuda.max_memory_allocated() / 1e9)
    del model, enc
    free()

    # 3. static cache + CUDA graphs.
    #
    # Do NOT hand-roll this as torch.compile(model.forward, mode="reduce-overhead").
    # transformers already does it, better, via generate()'s auto-compile path:
    # `_valid_auto_compile_criteria` turns it on for any compilable cache on CUDA,
    # and CompileConfig defaults to mode="reduce-overhead" -- i.e. CUDA graphs.
    # Crucially it compiles DECODE ONLY (see `prefill_consumed` in generation/
    # utils.py), leaving prefill eager.
    #
    # That split is load-bearing. StaticLayer.lazy_initialization allocates the KV
    # tensors on the first forward and calls torch._dynamo.mark_static_address on
    # them, but guards it with `if not is_torchdynamo_compiling()`. Compiling the
    # whole forward means the cache is born mid-trace, the marking is skipped, and
    # inductor then bails with "skipping cudagraphs due to mutated inputs (84
    # instances)" -- exactly 28 layers x (keys, values, cumulative_length). So the
    # hand-rolled version silently gets NO cuda graphs, and on torch 2.9.1 it goes
    # on to segfault (SIGSEGV, exit 139) during decode.
    #
    # Passing compile_config explicitly rather than relying on the default, so the
    # mode this row claims to measure is visible at the call site.
    model = load()
    enc = make_batch(tokenizer, 1, args.prompt_tokens)
    t0 = time.perf_counter()
    ttft, dec, _ = measure(model, enc, args.new_tokens, cache_implementation="static",
                           compile_config=CompileConfig(mode="reduce-overhead"))
    row("static + CUDA graphs", ttft, dec, torch.cuda.max_memory_allocated() / 1e9,
        f"(incl. {time.perf_counter() - t0:.0f}s compile)")
    del model, enc
    free()


def bench_roofline(tokenizer, args):
    """Decode is bandwidth bound: tok/s <= bandwidth / bytes-read-per-token."""
    model = load()
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    enc = make_batch(tokenizer, 1, args.prompt_tokens)
    _, dec, _ = measure(model, enc, args.new_tokens)

    ceiling = SPEC_BANDWIDTH_GBS * 1e9 / weights_bytes
    print(f"\n=== decode roofline (batch=1) ===")
    print(f"weights read per token : {weights_bytes / 1e9:.2f} GB")
    print(f"spec memory bandwidth  : {SPEC_BANDWIDTH_GBS:.0f} GB/s")
    print(f"theoretical ceiling    : {ceiling:.0f} tok/s")
    print(f"measured               : {dec:.0f} tok/s")
    print(f"bandwidth utilisation  : {dec / ceiling * 100:.0f}%")
    del model, enc
    free()


BENCHES = {
    "dtype": bench_dtype,
    "attn": bench_attn,
    "batch": bench_batch,
    "compile": bench_compile,
    "roofline": bench_roofline,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=[*BENCHES, "all"])
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"prompt={args.prompt_tokens} new={args.new_tokens}")

    for name in (BENCHES if args.which == "all" else [args.which]):
        BENCHES[name](tok, args)
