"""DFlash minimal proof-of-concept.

Demonstrates the paper's core claim on the smallest released Transformers-backend
config: a lightweight block-diffusion draft model gives lossless multi-token
acceleration over plain autoregressive decoding.

Target : Qwen/Qwen3-4B
Draft  : z-lab/Qwen3-4B-DFlash-b16
Dataset: gsm8k (a few prompts)

For each prompt we decode the SAME target model under a configurable sweep of
inference-time draft block sizes (``DFLASH_BLOCK_SIZES``, default
``"1,4,8,12,16"``). ``block_size=1`` is plain autoregressive greedy decoding
and acts as the baseline; the other sizes invoke block-diffusion speculative
drafting + verify against the same b16 draft. The verifier accepts any
``block_size <= draft.block_size``, so this sweep characterises the throughput
vs. acceptance-length curve and identifies the practical sweet spot rather
than only the trained default of 16. We report per-block-size decode speedup,
mean acceptance length, and a losslessness check (under greedy decoding every
DFlash output must match the baseline output token for token).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.benchmark import _apply_chat_template, load_and_process_dataset
from dflash.model import DFlashDraftModel, dflash_generate

MODEL = os.environ.get("DFLASH_MODEL", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3-4B-DFlash-b16")
DATASET = os.environ.get("DFLASH_DATASET", "gsm8k")
MAX_SAMPLES = int(os.environ.get("DFLASH_MAX_SAMPLES", "20"))
MAX_NEW_TOKENS = int(os.environ.get("DFLASH_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("DFLASH_TEMPERATURE", "0.0"))
BLOCK_SIZES = [
    int(s) for s in os.environ.get("DFLASH_BLOCK_SIZES", "1,4,8,12,16").split(",") if s.strip()
]

ART = Path(".openresearch/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def _attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        logger.warning("flash_attn not installed; using sdpa (lower absolute speedup).")
        return "sdpa"


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    attn = _attn_impl()
    logger.info(f"attn_implementation={attn}")

    logger.info(f"Loading target {MODEL} ...")
    target = (
        AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation=attn, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    logger.info(f"Loading draft {DRAFT} ...")
    draft = (
        DFlashDraftModel.from_pretrained(DRAFT, attn_implementation=attn, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    draft_block_size = draft.block_size
    logger.info(f"draft.block_size={draft_block_size}  target_layer_ids={draft.target_layer_ids}")

    # Validate sweep against the draft's maximum block size and ensure baseline=1
    # is included so we have something to measure speedups against.
    if not BLOCK_SIZES:
        raise ValueError("DFLASH_BLOCK_SIZES must contain at least one positive integer")
    for bs in BLOCK_SIZES:
        if bs < 1:
            raise ValueError(f"block_size must be >= 1, got {bs}")
        if bs > draft_block_size:
            raise ValueError(
                f"block_size={bs} exceeds draft.block_size={draft_block_size}; "
                f"dflash_generate only accepts block_size <= model.block_size"
            )
    if 1 not in BLOCK_SIZES:
        raise ValueError("DFLASH_BLOCK_SIZES must include 1 (used as the AR baseline)")
    logger.info(f"Sweeping block_sizes={BLOCK_SIZES} against draft b{draft_block_size}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = load_and_process_dataset(DATASET)[:MAX_SAMPLES]

    # Warmup (compile/caches) on a short prompt so timings are fair.
    warm = tokenizer.encode(
        _apply_chat_template(tokenizer, [{"role": "user", "content": "Hi"}], False),
        return_tensors="pt",
    ).to(device)
    for bs in BLOCK_SIZES:
        dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                        stop_token_ids=[tokenizer.eos_token_id], temperature=TEMPERATURE,
                        block_size=bs, return_stats=True)

    rows = []
    for i, instance in enumerate(dataset):
        prompt = instance["turns"][0]
        text = _apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], False)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        results = {}
        for bs in BLOCK_SIZES:
            results[bs] = dflash_generate(
                draft, target=target, input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS, stop_token_ids=[tokenizer.eos_token_id],
                temperature=TEMPERATURE, block_size=bs, return_stats=True,
            )

        base = results[1]
        base_ids = base.output_ids[0, base.num_input_tokens:].tolist()
        base_tpot = base.time_per_output_token

        row = {"idx": i, "baseline_tokens": len(base_ids)}
        for bs in BLOCK_SIZES:
            spec = results[bs]
            spec_ids = spec.output_ids[0, spec.num_input_tokens:].tolist()
            # Longest common prefix as a losslessness probe (greedy => should match).
            lcp = 0
            for a, b in zip(base_ids, spec_ids):
                if a != b:
                    break
                lcp += 1
            lossless = base_ids == spec_ids
            speedup = base_tpot / spec.time_per_output_token
            mean_acc = float(np.mean(spec.acceptance_lengths)) if spec.acceptance_lengths else 0.0
            row[f"bs{bs}"] = {
                "tpot_ms": round(spec.time_per_output_token * 1e3, 3),
                "decode_speedup": round(speedup, 3),
                "mean_acceptance_length": round(mean_acc, 3),
                "tokens": len(spec_ids),
                "lossless": lossless,
                "lcp_over_min_len": round(lcp / max(1, min(len(base_ids), len(spec_ids))), 4),
            }
        rows.append(row)
        summary = "  ".join(
            f"b{bs}:{row[f'bs{bs}']['decode_speedup']:.2f}x/acc{row[f'bs{bs}']['mean_acceptance_length']:.2f}"
            for bs in BLOCK_SIZES
        )
        logger.info(f"[{i}] {summary}")

    with open(ART / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Aggregate per block size across samples.
    sweep = []
    for bs in BLOCK_SIZES:
        tpots = [r[f"bs{bs}"]["tpot_ms"] for r in rows]
        accs = [r[f"bs{bs}"]["mean_acceptance_length"] for r in rows]
        losslessness = [1.0 if r[f"bs{bs}"]["lossless"] else 0.0 for r in rows]
        mean_tpot = float(np.mean(tpots))
        sweep.append({
            "block_size": bs,
            "tpot_ms": mean_tpot,
            "tps": 1e3 / mean_tpot,
            "mean_acceptance_length": float(np.mean(accs)),
            "lossless_frac": float(np.mean(losslessness)),
        })
    baseline = next(s for s in sweep if s["block_size"] == 1)
    for s in sweep:
        s["decode_speedup"] = baseline["tpot_ms"] / s["tpot_ms"]

    best = max(sweep, key=lambda s: s["decode_speedup"])

    table_rows = "\n".join(
        f"| {s['block_size']} | {s['tps']:.1f} | {s['decode_speedup']:.2f}x | "
        f"{s['mean_acceptance_length']:.2f} | {s['lossless_frac'] * 100:.0f}% |"
        for s in sweep
    )

    md = f"""# DFlash PoC: {MODEL} + {DRAFT}

Backend: Transformers ({attn}) | Dataset: {DATASET} | Samples: {len(rows)} | \
draft.block_size: {draft_block_size} | temperature: {TEMPERATURE} | max_new_tokens: {MAX_NEW_TOKENS}

## Inference-time block-size sweep

`dflash_generate` accepts any `block_size <= draft.block_size` against the same
b{draft_block_size} draft. ``block_size=1`` is plain autoregressive decoding
(baseline); larger sizes propose more tokens per verify step. The sweep below
characterises the throughput vs. acceptance-length curve and identifies the
practical sweet spot.

| draft block_size | throughput (tok/s) | decode speedup vs. AR | mean acceptance length | lossless |
|---|---|---|---|---|
{table_rows}

**Best speedup**: block_size={best['block_size']} at {best['decode_speedup']:.2f}x \
(mean acceptance length {best['mean_acceptance_length']:.2f}).

- Both decoders use the same frozen target {MODEL}; only the draft block size differs.
- Acceptance length = mean tokens accepted per target forward pass. >1 means
  the block-diffusion draft proposed multiple correct tokens at once. The
  upper bound at block_size=k is k.
- Under greedy decoding DFlash is lossless by construction (verify step), so
  every DFlash output should match the baseline output token for token.

Per-sample numbers in `samples.jsonl`.
"""
    (ART / "EVAL.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
