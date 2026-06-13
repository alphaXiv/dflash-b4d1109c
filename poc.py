"""DFlash minimal proof-of-concept.

Demonstrates the paper's core claim on the smallest released Transformers-backend
config: a lightweight block-diffusion draft model gives lossless multi-token
acceleration over plain autoregressive decoding.

Target  : Qwen/Qwen3-4B
Draft   : z-lab/Qwen3-4B-DFlash-b16
Datasets: gsm8k, humaneval, mt-bench (a few prompts each)

For each prompt we decode twice with the SAME target model:
  * baseline  : block_size=1  (ordinary autoregressive greedy decoding)
  * dflash    : block_size=16 (block-diffusion speculative drafting + verify)
We report decode speedup, mean acceptance length, and a losslessness check
(under greedy decoding the DFlash output must match the baseline output token
for token, since every drafted token is verified by the target).
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
_DEFAULT_DATASETS = "gsm8k,humaneval,mt-bench"
DATASETS = [d.strip() for d in os.environ.get("DFLASH_DATASETS", _DEFAULT_DATASETS).split(",") if d.strip()]
MAX_SAMPLES = int(os.environ.get("DFLASH_MAX_SAMPLES", "20"))
MAX_NEW_TOKENS = int(os.environ.get("DFLASH_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("DFLASH_TEMPERATURE", "0.0"))

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
    block_size = draft.block_size
    logger.info(f"draft.block_size={block_size}  target_layer_ids={draft.target_layer_ids}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Warmup (compile/caches) on a short prompt so timings are fair.
    warm = tokenizer.encode(
        _apply_chat_template(tokenizer, [{"role": "user", "content": "Hi"}], False),
        return_tensors="pt",
    ).to(device)
    for bs in (1, block_size):
        dflash_generate(draft, target=target, input_ids=warm, max_new_tokens=8,
                        stop_token_ids=[tokenizer.eos_token_id], temperature=TEMPERATURE,
                        block_size=bs, return_stats=True)

    all_rows = []  # flat list, every entry tagged with its dataset
    dataset_summaries = []  # one entry per dataset
    for ds_name in DATASETS:
        logger.info(f"=== Dataset: {ds_name} ===")
        dataset = load_and_process_dataset(ds_name)[:MAX_SAMPLES]

        rows = []
        for i, instance in enumerate(dataset):
            prompt = instance["turns"][0]
            text = _apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], False)
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

            base = dflash_generate(draft, target=target, input_ids=input_ids,
                                   max_new_tokens=MAX_NEW_TOKENS, stop_token_ids=[tokenizer.eos_token_id],
                                   temperature=TEMPERATURE, block_size=1, return_stats=True)
            spec = dflash_generate(draft, target=target, input_ids=input_ids,
                                   max_new_tokens=MAX_NEW_TOKENS, stop_token_ids=[tokenizer.eos_token_id],
                                   temperature=TEMPERATURE, block_size=block_size, return_stats=True)

            base_ids = base.output_ids[0, base.num_input_tokens:].tolist()
            spec_ids = spec.output_ids[0, spec.num_input_tokens:].tolist()
            # Longest common prefix as a losslessness probe (greedy => should be full match).
            lcp = 0
            for a, b in zip(base_ids, spec_ids):
                if a != b:
                    break
                lcp += 1
            lossless = base_ids == spec_ids

            speedup = base.time_per_output_token / spec.time_per_output_token
            mean_acc = float(np.mean(spec.acceptance_lengths))
            row = {
                "dataset": ds_name,
                "idx": i,
                "baseline_tpot_ms": round(base.time_per_output_token * 1e3, 3),
                "dflash_tpot_ms": round(spec.time_per_output_token * 1e3, 3),
                "decode_speedup": round(speedup, 3),
                "mean_acceptance_length": round(mean_acc, 3),
                "baseline_tokens": len(base_ids),
                "dflash_tokens": len(spec_ids),
                "lossless": lossless,
                "lcp_over_min_len": round(lcp / max(1, min(len(base_ids), len(spec_ids))), 4),
            }
            rows.append(row)
            all_rows.append(row)
            logger.info(f"[{ds_name} {i}] speedup={speedup:.2f}x  acc_len={mean_acc:.2f}  lossless={lossless}")

        base_tpot = float(np.mean([r["baseline_tpot_ms"] for r in rows]))
        spec_tpot = float(np.mean([r["dflash_tpot_ms"] for r in rows]))
        ds_speedup = base_tpot / spec_tpot
        ds_mean_acc = float(np.mean([r["mean_acceptance_length"] for r in rows]))
        ds_lossless_frac = float(np.mean([1.0 if r["lossless"] else 0.0 for r in rows]))
        dataset_summaries.append({
            "dataset": ds_name,
            "samples": len(rows),
            "baseline_tps": 1e3 / base_tpot,
            "dflash_tps": 1e3 / spec_tpot,
            "speedup": ds_speedup,
            "mean_acceptance_length": ds_mean_acc,
            "lossless_frac": ds_lossless_frac,
        })

    with open(ART / "samples.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    header = (
        f"# DFlash PoC: {MODEL} + {DRAFT}\n\n"
        f"Backend: Transformers ({attn}) | Datasets: {', '.join(DATASETS)} | "
        f"Samples/dataset: {MAX_SAMPLES} | block_size: {block_size} | "
        f"temperature: {TEMPERATURE} | max_new_tokens: {MAX_NEW_TOKENS}\n\n"
        f"## Per-dataset: speedup and mean acceptance length\n\n"
        f"| Dataset | Samples | Baseline tok/s | DFlash tok/s | Decode speedup | "
        f"Mean acceptance length (of {block_size}) | Lossless |\n"
        f"|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for s in dataset_summaries:
        body += (
            f"| {s['dataset']} | {s['samples']} | {s['baseline_tps']:.1f} | "
            f"{s['dflash_tps']:.1f} | **{s['speedup']:.2f}x** | "
            f"**{s['mean_acceptance_length']:.2f}** | {s['lossless_frac'] * 100:.0f}% |\n"
        )
    notes = (
        "\n- Both decoders use the same frozen target "
        f"{MODEL}; only the drafting differs.\n"
        "- Acceptance length = mean tokens accepted per target forward pass. >1 means\n"
        "  the block-diffusion draft proposed multiple correct tokens at once.\n"
        "- Acceptance length is highly task-dependent (code vs. math vs. open chat),\n"
        "  so this row-per-dataset view characterizes how the b16 Qwen3-4B draft\n"
        "  generalizes beyond a single math benchmark.\n"
        "- Under greedy decoding DFlash is lossless by construction (verify step), so\n"
        "  the DFlash output should match the baseline output token for token.\n"
        "\nPer-sample numbers in `samples.jsonl` (tagged with `dataset`).\n"
    )
    md = header + body + notes
    (ART / "EVAL.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
