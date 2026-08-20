#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${IMAGE:-}" ]]; then
  echo "Set IMAGE to a registry path, for example:" >&2
  echo "  IMAGE=registry.example.com/rs/geochangemllm:0.1.0 bash scripts/build_push_image.sh" >&2
  exit 2
fi

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${IMAGE}" \
  .
docker push "${IMAGE}"
