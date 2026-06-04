#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/2dgs_aigc/configs/objectB.yaml"

# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_threestudio
apply_cuda_from_config "$CFG" "$ROOT"

export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.text_to_3d.run_objectB --config "$CFG"
