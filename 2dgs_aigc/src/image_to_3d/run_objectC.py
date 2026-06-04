from __future__ import annotations

import argparse

from .pipeline import run_image_to_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2dgs_aigc/configs/objectC.yaml")
    args = ap.parse_args()
    run_image_to_3d(args.config)


if __name__ == "__main__":
    main()

