from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.utils.cuda import build_cuda_env, get_cuda_config
from src.utils.config import get_wandb_config, load_yaml
from src.utils.paths import abs_path, ensure_dir
from src.utils.runner import run_cmd
from src.utils.template import render_template
from src.utils.wandb_utils import init_wandb, wandb_log


def _mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {}) or {}
    m = cfg.get("magic123", {}) or {}
    seed = cfg.get("seed", 0)
    wname = (cfg.get("wandb", {}) or {}).get("name", None) or "objectC"
    cc = get_cuda_config(cfg)
    return {
        "seed": seed,
        "wandb_name": wname,
        "cuda_enable": cc.enable,
        "cuda_device_ids": cc.device_ids,
        **paths,
        **m,
    }


def run_image_to_3d(config_path: str | Path) -> None:
    cfg_path = abs_path(config_path)
    cfg = load_yaml(cfg_path)
    os.environ.update(build_cuda_env(cfg))
    cc = get_cuda_config(cfg)
    wcfg = get_wandb_config(cfg)
    run = init_wandb(wcfg, config={"config_path": str(cfg_path), **cfg})
    wandb_log(run, {"cuda_enable": cc.enable, "cuda_device_ids": cc.device_ids})

    mapping = _mapping(cfg)
    logs_root = abs_path("2dgs_aigc/logs")
    log_dir = ensure_dir(logs_root / (wcfg.name or "objectC"))

    m = cfg.get("magic123", {}) or {}
    project_root = m.get("project_root", None)
    cwd = abs_path(project_root) if project_root else None

    t0 = time.time()
    wandb_log(run, {"stage": "magic123_run"})
    for i, raw in enumerate(m.get("run_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"magic123_run_{i:02d}", check=True)

    wandb_log(run, {"stage": "magic123_export"})
    for i, raw in enumerate(m.get("export_mesh_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"magic123_export_{i:02d}", check=True)

    wandb_log(run, {"elapsed_s": time.time() - t0, "status": "done"})
    if run is not None:
        run.finish()

