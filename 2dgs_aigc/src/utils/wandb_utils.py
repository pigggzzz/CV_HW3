from __future__ import annotations

from typing import Any, Mapping

from .config import WandbConfig


def init_wandb(wcfg: WandbConfig, *, config: Mapping[str, Any] | None = None):
    if not wcfg.enable or wcfg.mode == "disabled":
        return None
    import wandb  # local import to avoid hard dependency at import time

    return wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        name=wcfg.name,
        tags=wcfg.tags,
        mode=wcfg.mode,
        config=dict(config or {}),
    )


def wandb_log(run, metrics: Mapping[str, Any], *, step: int | None = None):
    if run is None:
        return
    run.log(dict(metrics), step=step)

