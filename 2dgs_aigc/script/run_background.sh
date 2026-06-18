#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/2dgs_aigc/configs/background.yaml"

# 背景场景仅 2DGS，无 COLMAP
# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_gs
apply_cuda_from_config "$CFG" "$ROOT"

export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.reconstruction.run_background --config "$CFG" --stage gs
