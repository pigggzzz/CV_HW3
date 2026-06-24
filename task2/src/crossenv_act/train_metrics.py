from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
KV_RE = re.compile(
    r"(?P<key>[A-Za-z_./-]+):\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[KMBkmb]?)"
)

KEY_ALIASES = {
    "step": "steps",
    "smpl": "samples",
    "ep": "episodes",
    "epch": "epochs",
    "loss": "train_loss",
    "grdn": "grad_norm",
    "lr": "lr",
    "updt_s": "update_s",
    "data_s": "dataloading_s",
    "smp/s": "samples_per_s",
    "mem_gb": "gpu_mem_gb",
}

# Metrics we most care about for the report. Other numeric wandb summary keys are
# kept only if they map to these names, to avoid CSVs full of internal wandb fields.
SUMMARY_KEY_ALIASES = {
    "loss": "train_loss",
    "train/loss": "train_loss",
    "train_loss": "train_loss",
    "l1_loss": "action_l1_loss",
    "train/l1_loss": "action_l1_loss",
    "action_l1_loss": "action_l1_loss",
    "kld_loss": "kld_loss",
    "train/kld_loss": "kld_loss",
    "grad_norm": "grad_norm",
    "train/grad_norm": "grad_norm",
    "lr": "lr",
    "train/lr": "lr",
    "update_s": "update_s",
    "dataloading_s": "dataloading_s",
    "samples_per_s": "samples_per_s",
    "gpu_mem_gb": "gpu_mem_gb",
    "steps": "steps",
    "samples": "samples",
    "episodes": "episodes",
    "epochs": "epochs",
}

PREFERRED_COLUMNS = [
    "time",
    "run_name",
    "mode",
    "status",
    "returncode",
    "elapsed_min",
    "steps_requested",
    "steps",
    "batch_size",
    "train_loss",
    "action_l1_loss",
    "kld_loss",
    "grad_norm",
    "lr",
    "samples_per_s",
    "gpu_mem_gb",
    "samples",
    "episodes",
    "epochs",
    "dataset_root",
    "run_dir",
    "log_path",
    "wandb_summary_path",
]


def strip_ansi(text: str) -> str:
    # tqdm sometimes leaves carriage-return updates. Treat them as independent lines.
    text = text.replace("\r", "\n")
    return ANSI_RE.sub("", text)


def parse_number(value: str) -> float | None:
    value = value.strip()
    multiplier = 1.0
    if value and value[-1] in "KkMmBb":
        suffix = value[-1].lower()
        value = value[:-1]
        multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}[suffix]
    try:
        return float(value) * multiplier
    except ValueError:
        return None


def parse_training_log(log_path: str | Path) -> dict[str, float]:
    path = Path(log_path)
    if not path.is_file():
        return {}
    metrics: dict[str, float] = {}
    # We keep the last occurrence of each metric, which corresponds to the final
    # LeRobot logging window. LeRobot's MetricsTracker prints strings like:
    # step:1K smpl:8K ep:4 epch:0.15 loss:0.123 grdn:1.23 lr:1.0e-05 ...
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_ansi(raw_line)
        if "loss:" not in line and "step:" not in line:
            continue
        found = False
        for match in KV_RE.finditer(line):
            key = match.group("key")
            out_key = KEY_ALIASES.get(key)
            if out_key is None:
                continue
            number = parse_number(match.group("value"))
            if number is None or not math.isfinite(number):
                continue
            metrics[out_key] = number
            found = True
        if found:
            metrics["_source_has_training_log_metrics"] = 1.0
    return metrics


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    else:
        out[prefix] = obj
    return out


def find_wandb_summary_for_run(output_dir: str | Path, run_name: str | None = None) -> Path | None:
    if not run_name:
        return None
    root = Path(output_dir)
    config_files = list(root.glob("wandb/**/files/config.yaml")) + list(root.glob("wandb/**/config.yaml"))
    matches: list[Path] = []
    for cfg in config_files:
        if not cfg.is_file():
            continue
        try:
            text = cfg.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if run_name in text:
            run_root = cfg.parent if cfg.parent.name != "files" else cfg.parent.parent
            candidates = [run_root / "files" / "wandb-summary.json", run_root / "wandb-summary.json"]
            matches.extend([c for c in candidates if c.is_file()])
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def find_latest_wandb_summary(output_dir: str | Path, start_epoch: float | None = None) -> Path | None:
    root = Path(output_dir)
    candidates = list(root.glob("wandb/**/wandb-summary.json")) + list(root.glob("wandb/**/files/wandb-summary.json"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    if start_epoch is not None:
        recent = [p for p in candidates if p.stat().st_mtime >= start_epoch - 60]
        if recent:
            candidates = recent
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_wandb_summary(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    flat = flatten_json(data)
    metrics: dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        simple_key = key.split("/")[-1]
        out_key = SUMMARY_KEY_ALIASES.get(key) or SUMMARY_KEY_ALIASES.get(simple_key)
        if out_key:
            metrics[out_key] = float(value)
    if metrics:
        metrics["_source_has_wandb_summary_metrics"] = 1.0
    return metrics


def csv_value(key: str, value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if key == "lr":
            return f"{value:.2e}"
        if key in {"steps", "samples", "episodes", "steps_requested", "batch_size", "returncode"}:
            return int(round(value))
        return f"{value:.2f}"
    return value


def normalized_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: csv_value(k, v) for k, v in row.items() if not k.startswith("_")}


def merge_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [k for k in PREFERRED_COLUMNS if k in keys]
    ordered.extend(sorted(k for k in keys if k not in ordered))
    return ordered


def append_history_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    rows.append(normalized_csv_row(row))
    fieldnames = merge_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_latest_csv(path: str | Path, row: dict[str, Any], key: str = "run_name") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    clean = normalized_csv_row(row)
    replaced = False
    for i, old in enumerate(rows):
        if old.get(key) == clean.get(key):
            old.update(clean)
            rows[i] = old
            replaced = True
            break
    if not replaced:
        rows.append(clean)
    fieldnames = merge_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_training_metrics_row(
    *,
    time_str: str,
    run_name: str,
    mode: str,
    status: str,
    returncode: int,
    elapsed_s: float,
    steps_requested: int,
    batch_size: int,
    dataset_root: str | Path,
    run_dir: str | Path,
    log_path: str | Path,
    output_dir: str | Path,
    start_epoch: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = find_wandb_summary_for_run(output_dir, run_name) or find_latest_wandb_summary(output_dir, start_epoch=start_epoch)
    log_metrics = parse_training_log(log_path)
    wandb_metrics = parse_wandb_summary(summary_path)
    metrics = {**log_metrics, **wandb_metrics}
    row: dict[str, Any] = {
        "time": time_str,
        "run_name": run_name,
        "mode": mode,
        "status": status,
        "returncode": returncode,
        "elapsed_min": elapsed_s / 60.0,
        "steps_requested": int(steps_requested),
        "batch_size": int(batch_size),
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "wandb_summary_path": str(summary_path) if summary_path else "",
        **{k: v for k, v in metrics.items() if not k.startswith("_")},
    }
    raw = {
        "row": row,
        "log_metrics": log_metrics,
        "wandb_metrics": wandb_metrics,
        "wandb_summary_path": str(summary_path) if summary_path else None,
    }
    return row, raw


def write_training_metric_outputs(output_dir: str | Path, run_dir: str | Path, row: dict[str, Any], raw: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-run copies.
    (run_dir / "training_metrics.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        clean = normalized_csv_row(row)
        fieldnames = merge_fieldnames([clean])
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(clean)
    # Global comparison tables for the report.
    update_latest_csv(output_dir / "training_metrics_latest.csv", row)
    append_history_csv(output_dir / "training_metrics_history.csv", row)
