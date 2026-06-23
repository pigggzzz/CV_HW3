#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="0"
DATA_DIR="/home/lama/task2/data"
OUTPUT_DIR="/home/lama/task2/output"
BASIC_POLICY=""
JOINT_POLICY=""
NUM_SEQUENCES="4"
SEQUENCE_IDS=""
START_INDICES=""
LIST_CANDIDATES_ONLY="false"
FAST_GENERATE="true"
STEPS_PER_SEQUENCE="32"
CANDIDATE_STRIDE="64"
MIN_FRAME_INDEX="5"
END_MARGIN="5"
MIN_VALID_HORIZON="32"
FPS="8"
WANDB_MODE="offline"
WANDB_ENABLE="true"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-id) CUDA_ID="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --basic-policy) BASIC_POLICY="$2"; shift 2 ;;
    --joint-policy) JOINT_POLICY="$2"; shift 2 ;;
    --num-sequences) NUM_SEQUENCES="$2"; shift 2 ;;
    --sequence-ids) SEQUENCE_IDS="$2"; shift 2 ;;
    --start-indices) START_INDICES="$2"; shift 2 ;;
    --list-candidates-only) LIST_CANDIDATES_ONLY="true"; shift ;;
    --fast-generate) FAST_GENERATE="true"; shift ;;
    --no-fast-generate) FAST_GENERATE="false"; shift ;;
    --steps-per-sequence) STEPS_PER_SEQUENCE="$2"; shift 2 ;;
    --candidate-stride) CANDIDATE_STRIDE="$2"; shift 2 ;;
    --min-frame-index) MIN_FRAME_INDEX="$2"; shift 2 ;;
    --end-margin) END_MARGIN="$2"; shift 2 ;;
    --min-valid-horizon) MIN_VALID_HORIZON="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --no-wandb) WANDB_ENABLE="false"; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$BASIC_POLICY" ]]; then
  BASIC_POLICY="$OUTPUT_DIR/act_env_b"
fi
if [[ -z "$JOINT_POLICY" ]]; then
  JOINT_POLICY="$OUTPUT_DIR/act_env_abc"
fi

WANDB_FLAG="--wandb-enable"
if [[ "$WANDB_ENABLE" == "false" ]]; then
  WANDB_FLAG="--no-wandb-enable"
fi

CMD=(
  python -m crossenv_act.offline_replay
  --cuda-id "$CUDA_ID"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --basic-policy "$BASIC_POLICY"
  --joint-policy "$JOINT_POLICY"
  --num-sequences "$NUM_SEQUENCES"
  --steps-per-sequence "$STEPS_PER_SEQUENCE"
  --candidate-stride "$CANDIDATE_STRIDE"
  --min-frame-index "$MIN_FRAME_INDEX"
  --end-margin "$END_MARGIN"
  --min-valid-horizon "$MIN_VALID_HORIZON"
  --fps "$FPS"
  --wandb-mode "$WANDB_MODE"
  "$WANDB_FLAG"
)

if [[ -n "$SEQUENCE_IDS" ]]; then
  CMD+=(--sequence-ids "$SEQUENCE_IDS")
fi
if [[ -n "$START_INDICES" ]]; then
  CMD+=(--start-indices "$START_INDICES")
fi
if [[ "$LIST_CANDIDATES_ONLY" == "true" ]]; then
  CMD+=(--list-candidates-only)
fi
if [[ "$FAST_GENERATE" == "true" ]]; then
  CMD+=(--fast-generate)
else
  CMD+=(--no-fast-generate)
fi

CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
