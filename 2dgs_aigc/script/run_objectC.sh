#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$ROOT/2dgs_aigc/configs/objectC.yaml"

# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_magic123
apply_cuda_from_config "$CFG" "$ROOT"

export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.image_to_3d.run_objectC --config "$CFG"
