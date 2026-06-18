#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_gs

export PYTHONPATH="$ROOT/2dgs_aigc:${PYTHONPATH:-}"
python -m src.fusion.export_assets \
  --objectA "$ROOT/2dgs_aigc/assets/meshes/objectA.ply" \
  --objectB "$ROOT/2dgs_aigc/assets/meshes/objectB.obj" \
  --objectC "$ROOT/2dgs_aigc/assets/meshes/objectC.obj" \
  --background "$ROOT/2dgs_aigc/assets/meshes/background.ply" \
  --out_dir "$ROOT/2dgs_aigc/assets/blender/meshes"
