#!/usr/bin/env bash
set -euo pipefail

echo "== Kubernetes nodes =="
kubectl get nodes -o wide \
  -L nvidia.com/gpu.sharing-strategy \
  -L nvidia.com/gpu.replicas

echo "== Allocatable NVIDIA GPUs =="
kubectl get nodes \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory'

echo "== NVIDIA device plugin / GPU operator pods =="
kubectl get pods -A \
  -l 'app.kubernetes.io/name in (nvidia-device-plugin,gpu-operator)'

echo "== Existing GPU requests =="
kubectl get pods -A \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,PHASE:.status.phase,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu,NODE:.spec.nodeName'
