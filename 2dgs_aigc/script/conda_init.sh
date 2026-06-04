#!/usr/bin/env bash
# 用法（必须 source）:
#   source 2dgs_aigc/script/conda_init.sh env_gs
#   source 2dgs_aigc/script/conda_init.sh env_gs /path/to/config.yaml

activate_conda_env() {
  local env_name="$1"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[conda_init] 未找到 conda，请先安装 Miniconda/Anaconda" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${env_name}"
}

apply_cuda_from_config() {
  local cfg_path="$1"
  local root="$2"
  if [[ ! -f "${cfg_path}" ]]; then
    echo "[conda_init] 配置文件不存在: ${cfg_path}" >&2
    return 1
  fi
  export PYTHONPATH="${root}/2dgs_aigc:${PYTHONPATH:-}"
  # 将 CUDA 环境变量注入当前 shell（供后续 python / 外部训练命令使用）
  # shellcheck disable=SC2046
  eval "$(python -m src.utils.cuda_env --config "${cfg_path}")"
}
