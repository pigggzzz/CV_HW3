from __future__ import annotations

import argparse

from .pipeline import run_text_to_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2dgs_aigc/configs/objectB.yaml")
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="跳过 SDI 训练，仅基于已有 checkpoint 高精度导出 mesh",
    )
    args = ap.parse_args()
    run_text_to_3d(args.config, export_only=args.export_only)


if __name__ == "__main__":
    main()

