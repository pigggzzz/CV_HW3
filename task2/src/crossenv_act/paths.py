from __future__ import annotations

from pathlib import Path

from .constants import WORK_DIR_NAME, RAW_DIR_NAME, PREPARED_DIR_NAME, JOINT_ABC_NAME


def resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def work_root(data_dir: str | Path) -> Path:
    return resolve_path(data_dir) / WORK_DIR_NAME


def raw_root(data_dir: str | Path) -> Path:
    return work_root(data_dir) / RAW_DIR_NAME


def prepared_root(data_dir: str | Path) -> Path:
    return work_root(data_dir) / PREPARED_DIR_NAME


def prepared_split_root(data_dir: str | Path, env: str) -> Path:
    return prepared_root(data_dir) / f"split{env.upper()}"


def prepared_joint_root(data_dir: str | Path) -> Path:
    return prepared_root(data_dir) / JOINT_ABC_NAME
