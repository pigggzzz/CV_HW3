from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .paths import resolve_path
from .train_metrics import build_training_metrics_row, write_training_metric_outputs


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _command_value(command: list[Any], prefix: str, default: Any = None) -> Any:
    for item in command:
        s = str(item)
        if s.startswith(prefix + "="):
            return s.split("=", 1)[1]
    return default


def summarize_entry(args) -> int:
    output_root = resolve_path(args.output_dir)
    run_names = args.run_name or ["act_env_b", "act_env_abc"]
    for run_name in run_names:
        run_dir = output_root / run_name
        meta = _read_json(run_dir / "run_metadata.json") or _read_json(output_root / "_metadata" / f"{run_name}.json")
        command = meta.get("command") or []
        mode = str(meta.get("mode") or ("ABC" if "abc" in run_name.lower() else "B"))
        dataset_root = meta.get("dataset_root") or ""
        steps = int(_command_value(command, "--steps", 0) or 0)
        batch_size = int(_command_value(command, "--batch_size", 0) or 0)
        log_path = output_root / "_logs" / f"{run_name}_train.log"
        row, raw = build_training_metrics_row(
            time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_name=run_name,
            mode=mode,
            status="summarized_existing_run",
            returncode=0,
            elapsed_s=0.0,
            steps_requested=steps,
            batch_size=batch_size,
            dataset_root=dataset_root,
            run_dir=run_dir,
            log_path=log_path,
            output_dir=output_root,
            start_epoch=None,
        )
        write_training_metric_outputs(output_root, run_dir, row, raw)
        tqdm.write(f"[summarize] {run_name}: wrote {run_dir / 'training_metrics.csv'}")
    tqdm.write(f"[summarize] comparison table: {output_root / 'training_metrics_latest.csv'}")
    return 0
