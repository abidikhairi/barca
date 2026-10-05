#!/usr/bin/env bash
# Offline tokenization / packing of a protein dataset.
#
#   ./scripts/preprocess.sh configs/lora.yaml --output /data/proteins/tokenized
set -euo pipefail

CONFIG="${1:-configs/lora.yaml}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"
"${PYTHON}" "${REPO_ROOT}/scripts/preprocess.py" --config "${CONFIG}" "$@"
