from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .data_prep import prepare_all, root_for_mode
from .paths import resolve_path, raw_root, prepared_root
from .train import write_json, runtime_env


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_count(root: Path) -> int:
    info = _read_json(root / "meta" / "info.json")
    return int(info.get("total_episodes", -1))


def _feature_summary(root: Path) -> dict:
    info = _read_json(root / "meta" / "info.json")
    features = info.get("features", {}) or {}
    return {
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "features": sorted(features.keys()),
        "has_action": "action" in features,
        "visual_features": sorted(k for k, ft in features.items() if isinstance(ft, dict) and str(ft.get("dtype", "")).lower() in {"image", "video"}),
    }


def inspect_entry(args) -> int:
    env = runtime_env(args.data_dir, args.output_dir, args.cuda_id)
    # No need to mutate os.environ for pure inspection except HF paths, but doing it keeps LeRobot consistent.
    import os
    os.environ.update(env)
    roots = prepare_all(
        args.data_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force_download=args.force_download,
        force_prepare=args.force_prepare,
    )
    output_root = resolve_path(args.output_dir)
    report = {
        "raw_root": str(raw_root(args.data_dir)),
        "prepared_root": str(prepared_root(args.data_dir)),
        "splits": {k: str(v) for k, v in roots.items()},
        "summary": {k: _feature_summary(v) for k, v in roots.items()},
    }
    out = output_root / "_metadata" / "prepared_dataset_report.json"
    write_json(out, report)
    tqdm.write(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    tqdm.write(f"[inspect] Wrote report to {out}")
    return 0
