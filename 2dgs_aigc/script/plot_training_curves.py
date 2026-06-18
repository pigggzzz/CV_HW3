#!/usr/bin/env python3
"""从已有训练记录绘制 Loss 曲线（CSV / shell 日志），供实验报告使用。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "2dgs_aigc"
LOGS = DATA / "2dgs_aigc" / "logs"
OUT_DEFAULT = ROOT / "latex" / "figures"


def load_pl_metrics_csv(path: Path) -> pd.DataFrame:
    """解析 PyTorch Lightning CSVLogger 导出的 metrics.csv（含交替空行）。"""
    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError(f"缺少 step 列: {path}")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    metric_cols = [c for c in df.columns if c.startswith("train/")]
    out = df[["step"] + metric_cols].dropna(subset=["step"])
    out = out.groupby("step", as_index=False).last().sort_values("step")
    return out


def parse_2dgs_log(path: Path, max_iter: int = 30000) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        rf"\|\s*(\d+)/{max_iter}[^\n]*?Loss=([0-9.]+)"
    )
    records: dict[int, float] = {}
    for m in pattern.finditer(text):
        records[int(m.group(1))] = float(m.group(2))
    if not records:
        raise ValueError(f"未从日志解析到 Loss: {path}")
    steps = sorted(records)
    return pd.DataFrame({"step": steps, "train/loss": [records[s] for s in steps]})


def plot_series(
    ax,
    df: pd.DataFrame,
    cols: list[str],
    title: str,
    ylabel: str = "Loss",
    smooth_window: int | None = None,
):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step / Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    for col in cols:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        x = df["step"]
        label = col.replace("train/", "")
        if smooth_window and len(y) > smooth_window:
            ys = y.rolling(smooth_window, min_periods=1).mean()
            ax.plot(x, ys, linewidth=1.2, label=label)
        else:
            ax.plot(x, y, linewidth=1.0, label=label)
    ax.legend(loc="upper right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
    })

    jobs: list[tuple[str, Path, list[str], int | None]] = [
        (
            "objectB_sdi",
            DATA / "data/processed/objectB/sdi_run/score-distillation-via-inversion/objectB/csv_logs/version_8/metrics.csv",
            ["train/loss_sdi", "train/loss_orient", "train/loss_sparsity"],
            50,
        ),
        (
            "objectC_coarse",
            DATA / "data/processed/objectC/magic123_run/magic123-coarse-sd/objectC/csv_logs/version_0/metrics.csv",
            ["train/loss_sd", "train/loss_sd_3d", "train/loss_rgb"],
            50,
        ),
        (
            "objectC_refine",
            DATA / "data/processed/objectC/magic123_run/magic123-refine-sd/objectC/csv_logs/version_2/metrics.csv",
            ["train/loss_sd", "train/loss_sd_3d", "train/loss_rgb", "train/loss_normal_consistency"],
            50,
        ),
    ]

    for name, csv_path, cols, smooth in jobs:
        if not csv_path.exists():
            print(f"[skip] 不存在: {csv_path}")
            continue
        df = load_pl_metrics_csv(csv_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_series(ax, df, cols, title=f"{name} — training loss", smooth_window=smooth)
        fig.tight_layout()
        out = out_dir / f"{name}_loss.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"[ok] {out}")

    log_jobs = [
        ("objectA_2dgs", LOGS / "objectA" / "20260606_081743__2dgs_train_00.log"),
        ("background_2dgs", LOGS / "background" / "20260607_041728__2dgs_train_00.log"),
    ]
    for name, log_path in log_jobs:
        if not log_path.exists():
            print(f"[skip] 不存在: {log_path}")
            continue
        df = parse_2dgs_log(log_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_series(ax, df, ["train/loss"], title=f"{name} — training loss", smooth_window=100)
        fig.tight_layout()
        out = out_dir / f"{name}_loss.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"[ok] {out}")

    # 总览拼图
    pngs = sorted(out_dir.glob("*_loss.png"))
    if len(pngs) >= 2:
        n = len(pngs)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, p in zip(axes_flat, pngs):
            ax.imshow(plt.imread(p))
            ax.set_title(p.stem.replace("_", " "), fontsize=9)
            ax.axis("off")
        for ax in axes_flat[len(pngs):]:
            ax.axis("off")
        fig.tight_layout()
        overview = out_dir / "all_training_loss_overview.png"
        fig.savefig(overview)
        plt.close(fig)
        print(f"[ok] {overview}")


if __name__ == "__main__":
    main()
