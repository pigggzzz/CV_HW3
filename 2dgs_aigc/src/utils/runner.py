from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .cuda import get_run_env
from .paths import ensure_dir


@dataclass(frozen=True)
class RunResult:
    cmd: str
    returncode: int
    elapsed_s: float
    log_path: Path | None = None


def run_cmd(
    cmd: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cfg: Mapping[str, Any] | None = None,
    log_dir: str | Path | None = None,
    name: str | None = None,
    check: bool = True,
) -> RunResult:
    start = time.time()
    log_path: Path | None = None

    stdout = None
    stderr = None
    if log_dir is not None:
        d = ensure_dir(log_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = (name or "run").replace("/", "_")
        log_path = d / f"{ts}__{safe}.log"
        f = log_path.open("w", encoding="utf-8")
        stdout = f
        stderr = subprocess.STDOUT
        f.write(f"$ {cmd}\n\n")
        f.flush()

    run_env = dict(env) if env is not None else get_run_env(cfg)

    try:
        proc = subprocess.run(
            shlex.split(cmd),
            cwd=str(cwd) if cwd is not None else None,
            env=run_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    finally:
        if log_path is not None:
            try:
                f.close()  # type: ignore[name-defined]
            except Exception:
                pass

    elapsed_s = time.time() - start
    res = RunResult(cmd=cmd, returncode=proc.returncode, elapsed_s=elapsed_s, log_path=log_path)
    if check and proc.returncode != 0:
        raise RuntimeError(f"命令执行失败 (code={proc.returncode}): {cmd}\nlog={log_path}")
    return res

