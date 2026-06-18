from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.utils.cuda import build_cuda_env, get_cuda_config
from src.utils.config import get_wandb_config, load_yaml
from src.utils.paths import abs_path, ensure_dir, resolve_work_dir
from src.utils.runner import run_cmd
from src.utils.template import render_template
from src.utils.wandb_utils import init_wandb, wandb_log


def _get_sdi_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """读取 SDI 配置段（兼容旧字段 threestudio）。"""
    sdi = cfg.get("sdi", {}) or cfg.get("threestudio", {}) or {}
    if not isinstance(sdi, dict):
        return {}
    return dict(sdi)


def _mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {}) or {}
    sdi = _get_sdi_cfg(cfg)
    seed = cfg.get("seed", 0)
    wname = (cfg.get("wandb", {}) or {}).get("name", None) or "objectB"

    prompt = sdi.get("prompt", None)
    prompt_file = paths.get("prompt_file", None)
    if prompt_file:
        pf = abs_path(prompt_file)
        if pf.exists():
            prompt = pf.read_text(encoding="utf-8").strip()
    if not prompt:
        prompt = "a 3d object"

    project_root = sdi.get("project_root", None)
    config_rel = sdi.get("config", "configs/sdi.yaml")
    config_path = str(abs_path(project_root) / config_rel) if project_root else config_rel

    work_dir = resolve_work_dir(sdi.get("work_dir"))
    mesh_out = str(abs_path(paths.get("mesh_out", "")))
    mesh_assets_dir = str(abs_path(paths.get("mesh_assets_dir", "")))

    cc = get_cuda_config(cfg)
    base = {
        "seed": seed,
        "wandb_name": wname,
        "config": config_path,
        "exp_name": sdi.get("exp_name", "score-distillation-via-inversion"),
        "gpu": sdi.get("gpu", "0"),
        "export_gpu": sdi.get("export_gpu", sdi.get("gpu", "0")),
        "cuda_enable": cc.enable,
        "cuda_device_ids": cc.device_ids,
        "isosurface_resolution": sdi.get("isosurface_resolution", 256),
        "isosurface_method": sdi.get("isosurface_method", "mc-cpu"),
        "isosurface_chunk": sdi.get("isosurface_chunk", 65536),
        "texture_size": sdi.get("texture_size", 2048),
        **paths,
        **sdi,
    }
    base["work_dir"] = work_dir
    base["mesh_out"] = mesh_out
    base["mesh_assets_dir"] = mesh_assets_dir
    base["prompt"] = prompt
    return base


def run_text_to_3d(config_path: str | Path, *, export_only: bool = False) -> None:
    cfg_path = abs_path(config_path)
    cfg = load_yaml(cfg_path)
    os.environ.update(build_cuda_env(cfg))
    cc = get_cuda_config(cfg)
    wcfg = get_wandb_config(cfg)
    run = init_wandb(wcfg, config={"config_path": str(cfg_path), **cfg})
    wandb_log(run, {"cuda_enable": cc.enable, "cuda_device_ids": cc.device_ids, "method": "sdi"})

    mapping = _mapping(cfg)
    logs_root = abs_path("2dgs_aigc/logs")
    log_dir = ensure_dir(logs_root / (wcfg.name or "objectB"))

    sdi = _get_sdi_cfg(cfg)
    project_root = sdi.get("project_root", None)
    cwd = abs_path(project_root) if project_root else None

    t0 = time.time()
    if not export_only:
        wandb_log(run, {"stage": "sdi_train"})
        for i, raw in enumerate(sdi.get("train_commands", []) or []):
            cmd = render_template(str(raw), mapping)
            run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"sdi_train_{i:02d}", check=True)

    wandb_log(run, {"stage": "sdi_export"})
    export_env = dict(os.environ)
    if cc.enable:
        export_env["CUDA_VISIBLE_DEVICES"] = str(sdi.get("export_gpu", sdi.get("gpu", "0")))
    else:
        export_env["CUDA_VISIBLE_DEVICES"] = ""
    for i, raw in enumerate(sdi.get("export_mesh_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"sdi_export_{i:02d}", check=True, env=export_env)

    wandb_log(run, {"elapsed_s": time.time() - t0, "status": "done"})
    if run is not None:
        run.finish()
