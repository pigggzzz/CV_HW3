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


def _get_magic123_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    m = cfg.get("magic123", {}) or {}
    if not isinstance(m, dict):
        return {}
    return dict(m)


def _mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {}) or {}
    m = _get_magic123_cfg(cfg)
    seed = cfg.get("seed", 0)
    wname = (cfg.get("wandb", {}) or {}).get("name", None) or "objectC"

    prompt = m.get("prompt", None)
    prompt_file = paths.get("prompt_file", None)
    if prompt_file:
        pf = abs_path(prompt_file)
        if pf.exists():
            prompt = pf.read_text(encoding="utf-8").strip()
    if not prompt:
        prompt = "a 3d object"

    input_image = str(abs_path(paths.get("input_image", "")))

    project_root = m.get("project_root", None)
    coarse_rel = m.get("coarse_config", "configs/magic123-coarse-sd.yaml")
    refine_rel = m.get("refine_config", "configs/magic123-refine-sd.yaml")
    coarse_config = str(abs_path(project_root) / coarse_rel) if project_root else coarse_rel
    refine_config = str(abs_path(project_root) / refine_rel) if project_root else refine_rel

    exp_name_coarse = m.get("exp_name_coarse", "magic123-coarse-sd")
    exp_name_refine = m.get("exp_name_refine", "magic123-refine-sd")
    work_dir = resolve_work_dir(m.get("work_dir"))
    coarse_ckpt = f"{work_dir}/{exp_name_coarse}/{wname}/ckpts/last.ckpt"

    threshold = m.get("isosurface_threshold", None)
    isosurface_threshold_arg = ""
    if threshold is not None:
        isosurface_threshold_arg = f"system.geometry_convert_override.isosurface_threshold={threshold}"

    mesh_out = str(abs_path(paths.get("mesh_out", "")))

    cc = get_cuda_config(cfg)
    base = {
        "seed": seed,
        "wandb_name": wname,
        "coarse_config": coarse_config,
        "refine_config": refine_config,
        "exp_name_coarse": exp_name_coarse,
        "exp_name_refine": exp_name_refine,
        "coarse_ckpt": coarse_ckpt,
        "isosurface_threshold_arg": isosurface_threshold_arg,
        "default_elevation_deg": m.get("default_elevation_deg", 0.0),
        "default_azimuth_deg": m.get("default_azimuth_deg", 0.0),
        "gpu": m.get("gpu", "0"),
        "wandb_project": m.get("wandb_project", (cfg.get("wandb", {}) or {}).get("project", "2dgs_aigc")),
        "wandb_run_coarse": m.get("wandb_run_coarse", f"{wname}-coarse"),
        "wandb_run_refine": m.get("wandb_run_refine", f"{wname}-refine"),
        "cuda_enable": cc.enable,
        "cuda_device_ids": cc.device_ids,
        **paths,
        **m,
    }
    base["work_dir"] = work_dir
    base["coarse_ckpt"] = coarse_ckpt
    base["prompt"] = prompt
    base["input_image"] = input_image
    base["mesh_out"] = mesh_out
    return base


def run_image_to_3d(config_path: str | Path) -> None:
    cfg_path = abs_path(config_path)
    cfg = load_yaml(cfg_path)
    os.environ.update(build_cuda_env(cfg))
    cc = get_cuda_config(cfg)
    wcfg = get_wandb_config(cfg)
    run = init_wandb(wcfg, config={"config_path": str(cfg_path), **cfg})
    wandb_log(run, {"cuda_enable": cc.enable, "cuda_device_ids": cc.device_ids, "method": "magic123"})

    mapping = _mapping(cfg)
    logs_root = abs_path("2dgs_aigc/logs")
    log_dir = ensure_dir(logs_root / (wcfg.name or "objectC"))

    m = _get_magic123_cfg(cfg)
    project_root = m.get("project_root", None)
    cwd = abs_path(project_root) if project_root else None

    train_cmds = m.get("train_commands", None) or m.get("run_commands", []) or []

    t0 = time.time()
    wandb_log(run, {"stage": "magic123_train"})
    for i, raw in enumerate(train_cmds):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"magic123_train_{i:02d}", check=True)

    wandb_log(run, {"stage": "magic123_export"})
    for i, raw in enumerate(m.get("export_mesh_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"magic123_export_{i:02d}", check=True)

    wandb_log(run, {"elapsed_s": time.time() - t0, "status": "done"})
    if run is not None:
        run.finish()
