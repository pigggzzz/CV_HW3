#!/usr/bin/env bash
set -euo pipefail

CUDA_ID=0
DATA_DIR="./data"
OUTPUT_DIR="./output"
WANDB_MODE="online"
WANDB_PROJECT="calvin-act-crossenv"
BASIC_POLICY=""
JOINT_POLICY=""
CALVIN_ROOT="${CALVIN_ROOT:-}"
CALVIN_DATASET_PATH=""
CALVIN_CONF_DIR=""
NUM_SEQUENCES=20
MAX_STEPS=360
MAX_SUBTASKS=5
RECORD_VIDEOS=5
VIDEO_FPS=15
VIDEO_EVERY_N=2
SHOW_GUI=0
ALLOW_UNPROCESSED=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-id) CUDA_ID="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --basic-policy) BASIC_POLICY="$2"; shift 2 ;;
    --joint-policy) JOINT_POLICY="$2"; shift 2 ;;
    --calvin-root) CALVIN_ROOT="$2"; shift 2 ;;
    --calvin-dataset-path) CALVIN_DATASET_PATH="$2"; shift 2 ;;
    --calvin-conf-dir) CALVIN_CONF_DIR="$2"; shift 2 ;;
    --num-sequences) NUM_SEQUENCES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-subtasks) MAX_SUBTASKS="$2"; shift 2 ;;
    --record-videos) RECORD_VIDEOS="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --video-every-n) VIDEO_EVERY_N="$2"; shift 2 ;;
    --show-gui) SHOW_GUI=1; shift ;;
    --allow-unprocessed-actions) ALLOW_UNPROCESSED=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$BASIC_POLICY" ]]; then BASIC_POLICY="$OUTPUT_DIR/act_env_b"; fi
if [[ -z "$JOINT_POLICY" ]]; then JOINT_POLICY="$OUTPUT_DIR/act_env_abc"; fi

CMD=(python -m crossenv_act sim-rollout
  --cuda-id "$CUDA_ID"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --basic-policy "$BASIC_POLICY"
  --joint-policy "$JOINT_POLICY"
  --num-sequences "$NUM_SEQUENCES"
  --max-steps "$MAX_STEPS"
  --max-subtasks "$MAX_SUBTASKS"
  --record-videos "$RECORD_VIDEOS"
  --video-fps "$VIDEO_FPS"
  --video-every-n "$VIDEO_EVERY_N"
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
)

if [[ -n "$CALVIN_ROOT" ]]; then CMD+=(--calvin-root "$CALVIN_ROOT"); fi
if [[ -n "$CALVIN_DATASET_PATH" ]]; then CMD+=(--calvin-dataset-path "$CALVIN_DATASET_PATH"); fi
if [[ -n "$CALVIN_CONF_DIR" ]]; then CMD+=(--calvin-conf-dir "$CALVIN_CONF_DIR"); fi
if [[ "$SHOW_GUI" == "1" ]]; then CMD+=(--show-gui); fi
if [[ "$ALLOW_UNPROCESSED" == "1" ]]; then CMD+=(--allow-unprocessed-actions); fi
CMD+=("${EXTRA[@]}")

printf '[sim_rollout.sh] Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
