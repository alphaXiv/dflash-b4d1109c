#!/usr/bin/env bash
# DFlash minimal proof-of-concept: Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16 on 1 GPU.
# Installs the Transformers backend, then runs poc.py which measures decode
# speedup, acceptance length, and losslessness, writing .openresearch/artifacts/.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "=== Installing DFlash (transformers backend) ==="
pip install -q -e ".[transformers]"
pip install -q hf_transfer || true

echo "=== Installing flash-attn (optional; falls back to sdpa) ==="
pip install -q flash-attn --no-build-isolation || \
  echo "flash-attn install failed; poc.py will use sdpa."

echo "=== Running PoC ==="
python poc.py
