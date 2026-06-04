from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_by_dotted_path(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _expand_vars(obj: Any, root: Mapping[str, Any]) -> Any:
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            try:
                return str(_get_by_dotted_path(root, key))
            except KeyError:
                return os.environ.get(key, match.group(0))

        out = obj
        for _ in range(10):
            new = _VAR_PATTERN.sub(repl, out)
            if new == out:
                break
            out = new
        return out
    if isinstance(obj, list):
        return [_expand_vars(x, root) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_vars(v, root) for k, v in obj.items()}
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 顶层必须是 dict: {p}")
    cfg = _expand_vars(cfg, cfg)
    return cfg


@dataclass(frozen=True)
class WandbConfig:
    enable: bool = True
    mode: str = "online"  # online/offline/disabled
    project: str = "2dgs_aigc"
    entity: str | None = None
    name: str | None = None
    tags: list[str] | None = None


def get_wandb_config(cfg: Mapping[str, Any]) -> WandbConfig:
    w = cfg.get("wandb", {}) if isinstance(cfg.get("wandb", {}), Mapping) else {}
    return WandbConfig(
        enable=bool(w.get("enable", True)),
        mode=str(w.get("mode", "online")),
        project=str(w.get("project", "2dgs_aigc")),
        entity=w.get("entity", None),
        name=w.get("name", None),
        tags=list(w.get("tags", [])) if w.get("tags", None) is not None else None,
    )

