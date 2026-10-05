#!/usr/bin/env bash
# Launch a Barca fine-tuning run.
#
#   ./scripts/train.sh configs/lora.yaml
#   ./scripts/train.sh configs/full_ft.yaml --output_dir runs/full-ft-2 --no_wandb
#   NUM_GPUS=4 ./scripts/train.sh configs/qlora.yaml
#
# Assumes the Python environment already provides torch/transformers/etc.
set -euo pipefail

CONFIG="${1:-configs/lora.yaml}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS="${NUM_GPUS:-1}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" -m barca.train --config "${CONFIG}" "$@"
else
  python -m barca.train --config "${CONFIG}" "$@"
fi
