"""DFlash minimal proof-of-concept.

Demonstrates the paper's core claim on the smallest released Transformers-backend
config: a lightweight block-diffusion draft model gives lossless multi-token
acceleration over plain autoregressive decoding.

Target : Qwen/Qwen3-4B
Draft  : z-lab/Qwen3-4B-DFlash-b16
Dataset: gsm8k (a few prompts)

For each prompt we decode once at baseline (block_size=1, ordinary autoregressive
greedy) and then sweep DFlash over block_size in [2, 4, 8, 16, 32] with the SAME
draft, so we can map the speedup-vs-losslessness Pareto as a function of the
verify-window size. For each block_size we report decode speedup, mean
acceptance length, per-token agreement vs the baseline, and whether the output
is bitwise identical to the baseline.
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
BLOCK_SIZES = [2, 4, 8, 16, 32]

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
    logger.info(f"draft.block_size={draft.block_size}  target_layer_ids={draft.target_layer_ids}  "
                f"sweep_block_sizes={BLOCK_SIZES}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = load_and_process_dataset(DATASET)[:MAX_SAMPLES]

    # Warmup (compile/caches) on a short prompt so timings are fair.
    warm = tokenizer.encode(
        _apply_chat_template(tokenizer, [{"role": "user", "content": "Hi"}], False),
        return_tensors="pt",
    ).to(device)
    for bs in [1, *BLOCK_SIZES]:
        dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                        stop_token_ids=[tokenizer.eos_token_id], temperature=TEMPERATURE,
                        block_size=bs, return_stats=True)

    rows = []
    for i, instance in enumerate(dataset):
        prompt = instance["turns"][0]
        text = _apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], False)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        base = dflash_generate(draft, target=target, input_ids=input_ids,
                               max_new_tokens=MAX_NEW_TOKENS, stop_token_ids=[tokenizer.eos_token_id],
                               temperature=TEMPERATURE, block_size=1, return_stats=True)
        base_ids = base.output_ids[0, base.num_input_tokens:].tolist()

        for bs in BLOCK_SIZES:
            spec = dflash_generate(draft, target=target, input_ids=input_ids,
                                   max_new_tokens=MAX_NEW_TOKENS,
                                   stop_token_ids=[tokenizer.eos_token_id],
                                   temperature=TEMPERATURE, block_size=bs, return_stats=True)
            spec_ids = spec.output_ids[0, spec.num_input_tokens:].tolist()
            n = min(len(base_ids), len(spec_ids))
            # Longest common prefix: how far DFlash tracks the token-by-token baseline.
            lcp = 0
            for a, b in zip(base_ids, spec_ids):
                if a != b:
                    break
                lcp += 1
            # Aligned per-token agreement over the shared length.
            agree = sum(1 for a, b in zip(base_ids[:n], spec_ids[:n]) if a == b)
            exact_match = base_ids == spec_ids
            first_div = lcp if lcp < n else (n if len(base_ids) == len(spec_ids) else n)

            speedup = base.time_per_output_token / spec.time_per_output_token
            mean_acc = float(np.mean(spec.acceptance_lengths))
            row = {
                "idx": i,
                "block_size": bs,
                "baseline_tpot_ms": round(base.time_per_output_token * 1e3, 3),
                "dflash_tpot_ms": round(spec.time_per_output_token * 1e3, 3),
                "decode_speedup": round(speedup, 3),
                "mean_acceptance_length": round(mean_acc, 3),
                "baseline_tokens": len(base_ids),
                "dflash_tokens": len(spec_ids),
                "exact_match": exact_match,
                "token_agreement": round(agree / max(1, n), 4),
                "lcp_frac": round(lcp / max(1, n), 4),
                "first_divergence_idx": first_div,
            }
            rows.append(row)
            logger.info(f"[{i}] bs={bs:>2}  speedup={speedup:.2f}x  acc_len={mean_acc:.2f}  "
                        f"token_agree={agree / max(1, n):.3f}  first_div={first_div}/{n}")

    with open(ART / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Per-block_size aggregation.
    base_tpot = float(np.mean([r["baseline_tpot_ms"] for r in rows if r["block_size"] == BLOCK_SIZES[0]]))
    base_tps = 1e3 / base_tpot

    summary = []
    for bs in BLOCK_SIZES:
        bs_rows = [r for r in rows if r["block_size"] == bs]
        spec_tpot = float(np.mean([r["dflash_tpot_ms"] for r in bs_rows]))
        speedup = base_tpot / spec_tpot
        mean_acc = float(np.mean([r["mean_acceptance_length"] for r in bs_rows]))
        exact_frac = float(np.mean([1.0 if r["exact_match"] else 0.0 for r in bs_rows]))
        mean_agree = float(np.mean([r["token_agreement"] for r in bs_rows]))
        mean_lcp = float(np.mean([r["lcp_frac"] for r in bs_rows]))
        spec_tps = 1e3 / spec_tpot
        summary.append({
            "block_size": bs,
            "dflash_tps": spec_tps,
            "decode_speedup": speedup,
            "mean_acceptance_length": mean_acc,
            "token_agreement": mean_agree,
            "lcp_frac": mean_lcp,
            "exact_match_frac": exact_frac,
        })

    with open(ART / "summary.jsonl", "w") as f:
        for s in summary:
            f.write(json.dumps(s) + "\n")

    header = ("| block_size | DFlash tok/s | Decode speedup | Mean acc. length | "
             "Token agreement | Common-prefix frac | Bitwise identical |\n"
             "|---|---|---|---|---|---|---|\n")
    table_rows = "\n".join(
        f"| {s['block_size']} | {s['dflash_tps']:.1f} | {s['decode_speedup']:.2f}x | "
        f"{s['mean_acceptance_length']:.2f} | {s['token_agreement'] * 100:.2f}% | "
        f"{s['lcp_frac'] * 100:.2f}% | {s['exact_match_frac'] * 100:.0f}% |"
        for s in summary
    )

    n_prompts = len({r["idx"] for r in rows})
    md = f"""# DFlash PoC: {MODEL} + {DRAFT}

Backend: Transformers ({attn}) | Dataset: {DATASET} | Prompts: {n_prompts} | \
block_size sweep: {BLOCK_SIZES} | temperature: {TEMPERATURE} | max_new_tokens: {MAX_NEW_TOKENS}

Baseline (AR, block_size=1) throughput: **{base_tps:.1f} tok/s**

## Speedup vs losslessness Pareto across block_size

{header}{table_rows}

- Both decoders use the same frozen target {MODEL} and the same draft {DRAFT}
  (trained with block_size=16); only the verify-window `block_size` passed to
  `dflash_generate` is swept.
- Acceptance length = mean tokens accepted per target forward pass. It is
  upper-bounded by `block_size`, so larger `block_size` can amortize more
  target forwards if the draft stays accurate.
- Losslessness: DFlash's verify step accepts a drafted token only if it equals
  the token the target itself would emit, so divergences vs the AR baseline
  come from batched-vs-sequential floating-point differences flipping an
  occasional argmax. This sweep shows how those divergences (token agreement,
  common-prefix fraction, bitwise-identical fraction) evolve with the verify
  window, giving a principled basis for choosing the default `block_size`.

Per-sample numbers in `samples.jsonl`; per-block_size aggregates in `summary.jsonl`.
"""
    (ART / "EVAL.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
