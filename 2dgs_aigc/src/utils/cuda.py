from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CudaConfig:
    enable: bool = True
    device_ids: str = "0"  # 例如 "0" 或 "0,1"


def get_cuda_config(cfg: Mapping[str, Any]) -> CudaConfig:
    c = cfg.get("cuda", {}) if isinstance(cfg.get("cuda", {}), Mapping) else {}
    ids = c.get("device_ids", c.get("device_id", "0"))
    if isinstance(ids, (list, tuple)):
        ids = ",".join(str(x) for x in ids)
    return CudaConfig(
        enable=bool(c.get("enable", True)),
        device_ids=str(ids),
    )


def build_cuda_env(cfg: Mapping[str, Any]) -> dict[str, str]:
    """
    根据配置生成子进程/当前 shell 可用的 CUDA 环境变量。
    - enable=true  : 设置 CUDA_VISIBLE_DEVICES 为指定 GPU
    - enable=false : 置空 CUDA_VISIBLE_DEVICES，屏蔽 GPU
    """
    cc = get_cuda_config(cfg)
    env: dict[str, str] = {}
    if cc.enable:
        env["CUDA_VISIBLE_DEVICES"] = cc.device_ids
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def get_run_env(cfg: Mapping[str, Any] | None = None) -> dict[str, str]:
    """合并当前环境变量与配置中的 CUDA 设置。"""
    env = dict(os.environ)
    if cfg is not None:
        env.update(build_cuda_env(cfg))
    return env
