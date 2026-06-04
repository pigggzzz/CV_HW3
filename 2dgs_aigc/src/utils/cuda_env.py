"""供 shell 脚本 eval 使用：根据 YAML 打印 export 语句。"""
from __future__ import annotations

import argparse

from src.utils.config import load_yaml
from src.utils.cuda import build_cuda_env
from src.utils.paths import abs_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(abs_path(args.config))
    for k, v in build_cuda_env(cfg).items():
        # shell-safe: 仅数字与逗号
        print(f'export {k}="{v}"')


if __name__ == "__main__":
    main()
