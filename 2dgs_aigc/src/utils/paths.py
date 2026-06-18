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


def resolve_work_dir(work_dir: str | None, default: str = "outputs") -> str:
    """
    threestudio 在 dependences/threestudio 下运行；相对路径会误写到
    dependences/threestudio/data/...，因此 work_dir 必须转成 2dgs_aigc 下的绝对路径。
    """
    return str(abs_path(work_dir or default))

