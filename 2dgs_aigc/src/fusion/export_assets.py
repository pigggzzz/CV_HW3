from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.paths import abs_path, ensure_dir


def publish_asset(in_path: Path, out_dir: Path, name: str) -> Path:
    """按源文件后缀原样复制到 Blender 资产目录（不伪造扩展名）。"""
    ensure_dir(out_dir)
    out_path = out_dir / f"{name}{in_path.suffix.lower()}"
    shutil.copy2(in_path, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectA", default="2dgs_aigc/assets/meshes/objectA.ply")
    ap.add_argument("--objectB", default="2dgs_aigc/assets/meshes/objectB.obj")
    ap.add_argument("--objectC", default="2dgs_aigc/assets/meshes/objectC.obj")
    ap.add_argument("--background", default="2dgs_aigc/assets/meshes/background.ply")
    ap.add_argument("--out_dir", default="2dgs_aigc/assets/blender/meshes")
    args = ap.parse_args()

    out_dir = ensure_dir(abs_path(args.out_dir))
    for name, p in [
        ("objectA", args.objectA),
        ("objectB", args.objectB),
        ("objectC", args.objectC),
        ("background", args.background),
    ]:
        in_path = abs_path(p)
        if not in_path.exists():
            raise FileNotFoundError(f"{name} mesh 不存在: {in_path}")
        out_path = publish_asset(in_path, out_dir, name)
        print(f"[export] {name}: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
