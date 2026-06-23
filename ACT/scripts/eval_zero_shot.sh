#!/usr/bin/env bash
set -euo pipefail

CUDA_ID=0
DATA_DIR=./data
OUTPUT_DIR=./output
WANDB_MODE=online
WANDB_ENABLE=true
BASIC_POLICY=""
JOINT_POLICY=""
BATCH_SIZE=8
NUM_WORKERS=4
MAX_BATCHES=""
MAX_SAMPLES=""
FORCE_DOWNLOAD=false
FORCE_PREPARE=false
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-id) CUDA_ID="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --no-wandb) WANDB_ENABLE=false; shift ;;
    --basic-policy) BASIC_POLICY="$2"; shift 2 ;;
    --joint-policy) JOINT_POLICY="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-batches) MAX_BATCHES="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --success-threshold)
      echo "[eval warning] --success-threshold is deprecated and ignored. This task now reports action-error metrics only." >&2
      shift 2 ;;
    --force-download) FORCE_DOWNLOAD=true; shift ;;
    --force-prepare|--force-conversion) FORCE_PREPARE=true; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$BASIC_POLICY" ]]; then
  BASIC_POLICY="$OUTPUT_DIR/act_env_b"
fi
if [[ -z "$JOINT_POLICY" ]]; then
  JOINT_POLICY="$OUTPUT_DIR/act_env_abc"
fi

CMD=(python -m crossenv_act eval
  --cuda-id "$CUDA_ID"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --wandb-mode "$WANDB_MODE"
  --basic-policy "$BASIC_POLICY"
  --joint-policy "$JOINT_POLICY"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS")

if [[ "$WANDB_ENABLE" == false ]]; then
  CMD+=(--no-wandb-enable)
fi
[[ -n "$MAX_BATCHES" ]] && CMD+=(--max-batches "$MAX_BATCHES")
[[ -n "$MAX_SAMPLES" ]] && CMD+=(--max-samples "$MAX_SAMPLES")
[[ "$FORCE_DOWNLOAD" == true ]] && CMD+=(--force-download)
[[ "$FORCE_PREPARE" == true ]] && CMD+=(--force-prepare)

CMD+=("${EXTRA[@]}")
exec "${CMD[@]}"
