from __future__ import annotations

import argparse

from .pipeline import run_reconstruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2dgs_aigc/configs/background.yaml")
    ap.add_argument(
        "--stage",
        default="gs",
        choices=["all", "colmap", "gs"],
        help="背景场景无 COLMAP，默认 gs",
    )
    args = ap.parse_args()
    run_reconstruction(args.config, stage=args.stage)


if __name__ == "__main__":
    main()

