#!/usr/bin/env bash
set -euo pipefail

python -m geochangemllm.make_smoke_data \
  --output-dir data/smoke \
  --num-samples 32 \
  --image-size 256

torchrun --standalone --nproc_per_node=2 -m geochangemllm.train \
  --manifest data/smoke/manifest.jsonl \
  --output-dir outputs/tiny-smoke \
  --text-model tiny \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 20 \
  --log-every 1

python -m geochangemllm.infer \
  --checkpoint outputs/tiny-smoke/checkpoint.pt \
  --manifest data/smoke/manifest.jsonl \
  --output-dir outputs/tiny-infer \
  --generate-text
