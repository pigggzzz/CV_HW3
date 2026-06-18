from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CondaEnvConfig:
    env_colmap: str = "env_colmap"
    env_gs: str = "env_gs"
    env_sdi: str = "env_sdi"
    env_magic123: str = "env_magic123"

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "CondaEnvConfig":
        c = cfg.get("conda", {}) if isinstance(cfg.get("conda", {}), Mapping) else {}
        # 兼容旧字段 env_name -> env_gs
        legacy_gs = c.get("env_name", None)
        return cls(
            env_colmap=str(c.get("env_colmap", "env_colmap")),
            env_gs=str(c.get("env_gs", legacy_gs or "env_gs")),
            env_sdi=str(c.get("env_sdi", c.get("env_threestudio", "env_sdi"))),
            env_magic123=str(c.get("env_magic123", "env_magic123")),
        )
