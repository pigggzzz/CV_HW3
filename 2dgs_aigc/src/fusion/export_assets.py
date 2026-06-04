from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from src.utils.paths import abs_path, ensure_dir


def convert_mesh(in_path: Path, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    mesh = trimesh.load(in_path, force="mesh")
    if mesh is None:
        raise RuntimeError(f"无法读取 mesh: {in_path}")
    mesh.export(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectA", default="2dgs_aigc/assets/meshes/objectA.obj")
    ap.add_argument("--objectB", default="2dgs_aigc/assets/meshes/objectB.obj")
    ap.add_argument("--objectC", default="2dgs_aigc/assets/meshes/objectC.obj")
    ap.add_argument("--background", default="2dgs_aigc/assets/meshes/background.obj")
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
        out_path = out_dir / f"{name}.obj"
        convert_mesh(in_path, out_path)
        print(f"[export] {name}: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()

