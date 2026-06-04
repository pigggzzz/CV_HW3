#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/2dgs_aigc/configs/fusion.yaml"

# 用 env_gs 里的 Python 读取 fusion.yaml 并设置 CUDA_VISIBLE_DEVICES（Blender 本身不依赖 conda）
# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_gs
apply_cuda_from_config "$CFG" "$ROOT"

BLENDER_BIN="${BLENDER_BIN:-blender}"
"$BLENDER_BIN" -b -P "$ROOT/2dgs_aigc/script/blender_fusion.py" -- --config "$CFG"
