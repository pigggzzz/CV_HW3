from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.reconstruction.colmap_utils import (
    normalize_undistorted_colmap_layout,
    reset_colmap_workspace,
    validate_colmap_result,
)
from src.utils.cuda import build_cuda_env, get_cuda_config
from src.utils.config import get_wandb_config, load_yaml
from src.utils.paths import abs_path, ensure_dir
from src.utils.runner import run_cmd
from src.utils.template import render_template
from src.utils.wandb_utils import init_wandb, wandb_log


@dataclass(frozen=True)
class ReconstructionJob:
    cfg_path: Path
    cfg: dict[str, Any]


def _common_mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {})
    external = cfg.get("external", {})
    colmap = cfg.get("colmap", {})
    gs = cfg.get("gs", {})
    seed = cfg.get("seed", 0)

    cc = get_cuda_config(cfg)
    mapping: dict[str, Any] = {
        "seed": seed,
        **paths,
        **external,
        **colmap,
        **gs,
        "wandb_name": (cfg.get("wandb", {}) or {}).get("name", None) or "run",
        "colmap_undistorted_dir": colmap.get("undistorted_dir", None),
        "cuda_enable": cc.enable,
        "cuda_device_ids": cc.device_ids,
    }
    return mapping


def _maybe_extract_frames(cfg: Mapping[str, Any], mapping: Mapping[str, Any], log_dir: Path):
    colmap = cfg.get("colmap", {}) or {}
    ef = (colmap.get("extract_frames", {}) or {}) if isinstance(colmap, Mapping) else {}
    if not ef.get("enable", False):
        return

    ffmpeg_bin = mapping.get("ffmpeg_bin", "ffmpeg")
    inp = render_template(str(ef["input_video"]), mapping)
    out_dir = abs_path(render_template(str(ef["out_dir"]), mapping))
    ensure_dir(out_dir)
    fps = int(ef.get("fps", 2))
    out_pattern = str(out_dir / "%06d.jpg")

    cmd = f"{ffmpeg_bin} -y -i {inp} -vf fps={fps} {out_pattern}"
    run_cmd(cmd, cfg=cfg, log_dir=log_dir, name="extract_frames", check=True)


def run_colmap(cfg: Mapping[str, Any], mapping: Mapping[str, Any], *, log_dir: Path):
    colmap = cfg.get("colmap", {}) or {}
    cmds = colmap.get("commands", []) or []
    if mapping.get("sparse_dir"):
        ensure_dir(mapping["sparse_dir"])
    if mapping.get("undistorted_dir"):
        ensure_dir(mapping["undistorted_dir"])

    for i, raw in enumerate(cmds):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, log_dir=log_dir, name=f"colmap_{i:02d}", check=True)


def run_2dgs_train(cfg: Mapping[str, Any], mapping: Mapping[str, Any], *, log_dir: Path):
    gs = cfg.get("gs", {}) or {}
    project_root = gs.get("project_root", None)
    cwd = abs_path(project_root) if project_root else None
    ensure_dir(abs_path(gs.get("work_dir", "data/processed/_gs")))
    for i, raw in enumerate(gs.get("train_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"2dgs_train_{i:02d}", check=True)


def run_2dgs_export_mesh(cfg: Mapping[str, Any], mapping: Mapping[str, Any], *, log_dir: Path):
    gs = cfg.get("gs", {}) or {}
    project_root = gs.get("project_root", None)
    cwd = abs_path(project_root) if project_root else None
    for i, raw in enumerate(gs.get("export_mesh_commands", []) or []):
        cmd = render_template(str(raw), mapping)
        run_cmd(cmd, cfg=cfg, cwd=cwd, log_dir=log_dir, name=f"2dgs_export_{i:02d}", check=True)


def _prepare_mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _common_mapping(cfg)
    gs = cfg.get("gs", {}) or {}
    if "iteration" not in mapping:
        mapping["iteration"] = gs.get("iteration", 30000)
    for k in list(mapping.keys()):
        if k.endswith("_dir") or k.endswith("_path") or k.endswith("_out") or k.endswith("_root"):
            v = mapping.get(k)
            if isinstance(v, str) and ("/" in v or v.endswith((".db", ".obj", ".ply", ".mp4"))):
                try:
                    mapping[k] = str(abs_path(v))
                except Exception:
                    pass
    return mapping


def run_reconstruction(config_path: str | Path, *, stage: str = "all") -> None:
    """
    stage:
      - colmap: 抽帧 + COLMAP（Object A）
      - gs:     2DGS 训练 + mesh 导出
      - all:    上述全部（需由 shell 分环境调用 colmap / gs 各一次）
    """
    cfg_path = abs_path(config_path)
    cfg = load_yaml(cfg_path)
    os.environ.update(build_cuda_env(cfg))
    cc = get_cuda_config(cfg)
    wcfg = get_wandb_config(cfg)
    run = init_wandb(wcfg, config={"config_path": str(cfg_path), "stage": stage, **cfg})
    wandb_log(run, {"cuda_enable": cc.enable, "cuda_device_ids": cc.device_ids, "pipeline_stage": stage})

    mapping = _prepare_mapping(cfg)
    logs_root = abs_path("2dgs_aigc/logs")
    job_name = wcfg.name or cfg_path.stem
    log_dir = ensure_dir(logs_root / job_name)

    t0 = time.time()
    has_colmap = bool(cfg.get("colmap"))

    if stage in ("all", "colmap"):
        if not has_colmap:
            raise ValueError(f"配置 {cfg_path} 无 colmap 段，不能使用 --stage colmap")
        wandb_log(run, {"stage": "reset_colmap_workspace"})
        reset_colmap_workspace(cfg, mapping)
        wandb_log(run, {"stage": "extract_frames"})
        _maybe_extract_frames(cfg, mapping, log_dir)
        wandb_log(run, {"stage": "colmap"})
        run_colmap(cfg, mapping, log_dir=log_dir)
        undist = mapping.get("colmap_undistorted_dir") or mapping.get("undistorted_dir")
        if undist:
            normalize_undistorted_colmap_layout(undist)
        colmap_stats = validate_colmap_result(
            cfg,
            sparse_root=Path(mapping.get("sparse_dir", "")),
            image_dir=mapping.get("image_dir"),
        )
        wandb_log(run, colmap_stats)
        if undist:
            undist_stats = validate_colmap_result(
                cfg,
                sparse_root=Path(undist) / "sparse",
                image_dir=Path(undist) / "images",
            )
            wandb_log(run, {f"undist_{k}": v for k, v in undist_stats.items()})

    if stage in ("all", "gs"):
        if has_colmap:
            undist = mapping.get("colmap_undistorted_dir") or mapping.get("undistorted_dir")
            if undist:
                normalize_undistorted_colmap_layout(undist)
                colmap_stats = validate_colmap_result(
                    cfg,
                    sparse_root=Path(undist) / "sparse",
                    image_dir=Path(undist) / "images",
                )
                wandb_log(run, colmap_stats)
        wandb_log(run, {"stage": "2dgs_train"})
        run_2dgs_train(cfg, mapping, log_dir=log_dir)
        wandb_log(run, {"stage": "2dgs_export"})
        run_2dgs_export_mesh(cfg, mapping, log_dir=log_dir)

    if stage not in ("all", "colmap", "gs"):
        raise ValueError(f"未知 stage: {stage}，应为 all|colmap|gs")

    wandb_log(run, {"elapsed_s": time.time() - t0, "status": "done", "pipeline_stage": stage})
    if run is not None:
        run.finish()

