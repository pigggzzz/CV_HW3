from __future__ import annotations

import argparse

from .pipeline import run_reconstruction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2dgs_aigc/configs/objectA.yaml")
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "colmap", "gs"],
        help="colmap=仅 COLMAP；gs=仅 2DGS；all=全流程（建议由 shell 分环境各跑 colmap/gs）",
    )
    args = ap.parse_args()
    run_reconstruction(args.config, stage=args.stage)


if __name__ == "__main__":
    main()

