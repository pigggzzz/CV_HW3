#!/usr/bin/env bash
set -euo pipefail

CUDA_ID=0
DATA_DIR=./data
OUTPUT_DIR=./output
WANDB_MODE=online
STEPS=100000
BATCH_SIZE=8
NUM_WORKERS=4
OVERWRITE=false
RESUME=false
FORCE_DOWNLOAD=false
FORCE_PREPARE=false
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-id) CUDA_ID="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --overwrite-output) OVERWRITE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --force-download) FORCE_DOWNLOAD=true; shift ;;
    --force-prepare|--force-conversion) FORCE_PREPARE=true; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(python -m crossenv_act train
  --mode ABC
  --run-name act_env_abc
  --cuda-id "$CUDA_ID"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --wandb-mode "$WANDB_MODE"
  --steps "$STEPS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS")

[[ "$OVERWRITE" == true ]] && CMD+=(--overwrite-output)
[[ "$RESUME" == true ]] && CMD+=(--resume)
[[ "$FORCE_DOWNLOAD" == true ]] && CMD+=(--force-download)
[[ "$FORCE_PREPARE" == true ]] && CMD+=(--force-prepare)

CMD+=("${EXTRA[@]}")
exec "${CMD[@]}"
