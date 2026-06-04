from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def abs_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root() / pp).resolve()


def ensure_dir(p: str | Path) -> Path:
    d = abs_path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

