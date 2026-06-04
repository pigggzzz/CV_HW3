#!/usr/bin/env bash
# 在 env_gs（仅 2DGS）中安装 CUDA 扩展与子模块依赖
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GS_ROOT="$ROOT/2dgs_aigc/dependences/2d-gaussian-splatting"

# shellcheck disable=SC1091
source "$ROOT/2dgs_aigc/script/conda_init.sh"
activate_conda_env env_gs

if [[ ! -d "$GS_ROOT" ]]; then
  echo "[install_2dgs] 未找到 2DGS 仓库: $GS_ROOT"
  exit 1
fi

cd "$GS_ROOT"
if [[ ! -d submodules/diff-surfel-rasterization ]]; then
  git submodule update --init --recursive
fi

pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn
pip install plyfile opencv-python open3d mediapy lpips scikit-image

echo "[install_2dgs] 完成。验证: python -c \"import diff_surfel_rasterization\""
