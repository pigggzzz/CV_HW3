#!/usr/bin/env bash

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/2dgs_aigc/configs/objectA.yaml"

# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"

# 阶段 1：COLMAP（env_colmap）
activate_conda_env env_colmap
apply_cuda_from_config "$CFG" "$ROOT"
export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.reconstruction.run_objectA --config "$CFG" --stage colmap

# 阶段 2：2DGS（env_gs）
activate_conda_env env_gs
apply_cuda_from_config "$CFG" "$ROOT"
export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.reconstruction.run_objectA --config "$CFG" --stage gs
