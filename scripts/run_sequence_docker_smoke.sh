#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-geochangemllm:0.2.0}"
DATA_VOLUME="${DATA_VOLUME:-geochangemllm-data}"
CACHE_VOLUME="${CACHE_VOLUME:-geochangemllm-cache}"
QWEN_PATH="${QWEN_PATH:-/cache/models/Qwen3-0.6B}"

docker run --rm \
  --name geochangemllm-sequence-smoke \
  --gpus all \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cpus=12 \
  --memory=48g \
  -e HOME=/home/trainer \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e OMP_NUM_THREADS=4 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IGNORE_DISABLED_P2P=1 \
  -e NCCL_IB_DISABLE=1 \
  -v "${DATA_VOLUME}:/data" \
  -v "${CACHE_VOLUME}:/cache" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail

    python -m geochangemllm.make_sequence_smoke_data \
      --output-dir /data/sequence-smoke \
      --num-sequences 8 \
      --frames-per-sequence 4 \
      --image-size 256

    torchrun --standalone --nproc_per_node=2 \
      -m geochangemllm.train_unsupervised \
      --manifest /data/sequence-smoke/manifest.jsonl \
      --output-dir /data/outputs/unsupervised-smoke \
      --pair-mode adjacent \
      --image-size 256 \
      --batch-size 2 \
      --epochs 20 \
      --max-steps 10 \
      --precision bf16 \
      --log-every 1

    torchrun --standalone --nproc_per_node=2 \
      -m geochangemllm.train_weak_sequence \
      --manifest /data/sequence-smoke/manifest.jsonl \
      --vision-checkpoint /data/outputs/unsupervised-smoke/vision_checkpoint.pt \
      --output-dir /data/outputs/weak-qwen-smoke \
      --text-model '${QWEN_PATH}' \
      --lora-rank 16 \
      --max-frames 4 \
      --max-text-tokens 256 \
      --image-size 256 \
      --batch-size 1 \
      --grad-accum 2 \
      --epochs 20 \
      --max-steps 10 \
      --precision bf16 \
      --log-every 1

    python -m geochangemllm.infer_sequence \
      --checkpoint /data/outputs/weak-qwen-smoke/checkpoint.pt \
      --manifest /data/sequence-smoke/manifest.jsonl \
      --output-dir /data/outputs/weak-qwen-infer \
      --generate-text
  "
