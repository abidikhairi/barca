#!/usr/bin/env bash
# Launch a Barca fine-tuning run.
#
#   ./scripts/train.sh configs/lora.yaml
#   ./scripts/train.sh configs/full_ft.yaml --output_dir runs/full-ft-2 --no_wandb
#   NUM_GPUS=4 ./scripts/train.sh configs/qlora.yaml
#
# Assumes the Python environment already provides torch/transformers/etc.
# Override the interpreter with PYTHON, e.g.:
#   PYTHON=/opt/miniconda3/bin/python ./scripts/train.sh configs/lora.yaml
set -euo pipefail

CONFIG="${1:-configs/lora.yaml}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Reduces CUDA allocator fragmentation for this attention-heavy model.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" -m barca.train --config "${CONFIG}" "$@"
else
  "${PYTHON}" -m barca.train --config "${CONFIG}" "$@"
fi
