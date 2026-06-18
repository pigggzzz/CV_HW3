from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Mapping

_COLMAP_MODEL_FILES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "points3D.ply",
    "frames.bin",
    "rigs.bin",
    "project.ini",
)


def count_registered_images(sparse_model_dir: str | Path) -> int:
    """读取 COLMAP sparse model 目录下 images.bin 中已注册图像数量。"""
    images_bin = Path(sparse_model_dir) / "images.bin"
    if not images_bin.exists():
        return 0
    with images_bin.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
    return int(num_images)


def count_input_images(image_dir: str | Path) -> int:
    d = Path(image_dir)
    if not d.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sum(1 for p in d.iterdir() if p.suffix in exts)


def resolve_sparse_model_dir(sparse_root: str | Path) -> Path | None:
    """
    定位 COLMAP sparse model 目录。
    - mapper 输出: sparse/0/
    - image_undistorter 输出: sparse/（文件直接在 sparse 下，无 0 子目录）
    """
    root = Path(sparse_root)
    candidates = [root / "0", root]
    for c in candidates:
        if (c / "images.bin").exists():
            return c
    return None


def normalize_undistorted_colmap_layout(undistorted_dir: str | Path) -> Path:
    """
    将 image_undistorter 产物的 sparse/*.bin 整理为 sparse/0/*.bin。
    2DGS 要求数据集根目录下存在 sparse/0/images.bin。
    """
    root = Path(undistorted_dir)
    sparse = root / "sparse"
    if not sparse.exists():
        raise FileNotFoundError(f"未找到 undistorted sparse 目录: {sparse}")

    model0 = sparse / "0"
    if (model0 / "images.bin").exists():
        return root

    if not (sparse / "images.bin").exists():
        resolved = resolve_sparse_model_dir(sparse)
        if resolved is None:
            raise FileNotFoundError(
                f"无法在 {sparse} 下找到 images.bin（既无 sparse/0 也无 sparse 根级文件）"
            )
        if resolved == model0:
            return root
        # 已解析到其它路径，不再移动
        return root

    model0.mkdir(parents=True, exist_ok=True)
    for name in _COLMAP_MODEL_FILES:
        src = sparse / name
        if not src.exists():
            continue
        dst = model0 / name
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
    return root


def reset_colmap_workspace(cfg: Mapping, mapping: Mapping | None = None) -> None:
    """清理 COLMAP 旧结果，避免 database 残留导致注册失败。"""
    colmap = cfg.get("colmap", {}) or {}
    if not colmap.get("reset_workspace", False):
        return

    src = mapping if mapping is not None else colmap
    for key in ("database_path", "sparse_dir", "undistorted_dir"):
        p = src.get(key)
        if not p:
            continue
        path = Path(p)
        if path.suffix == ".db" and path.exists():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def validate_colmap_result(
    cfg: Mapping,
    *,
    sparse_root: str | Path,
    image_dir: str | Path | None = None,
) -> dict[str, int]:
    colmap = cfg.get("colmap", {}) or {}
    min_reg = int(colmap.get("min_registered_images", 10))

    model_dir = resolve_sparse_model_dir(sparse_root)
    registered = count_registered_images(model_dir) if model_dir else 0
    input_n = count_input_images(image_dir) if image_dir else 0

    stats = {
        "colmap_input_images": input_n,
        "colmap_registered_images": registered,
        "colmap_min_required": min_reg,
        "colmap_sparse_model_dir": str(model_dir) if model_dir else None,
    }
    if registered < min_reg:
        raise RuntimeError(
            "COLMAP 注册视角过少，无法做可靠 2DGS 重建。"
            f" 已注册 {registered} 张，至少需要 {min_reg} 张"
            f"（输入目录共 {input_n} 张）。"
            f" 检查的 sparse 路径: {sparse_root}"
            f"（解析到: {model_dir}）。"
            " 若使用官方 COLMAP 数据集且注册数正常，可能是 undistort 后目录为 sparse/ 而非 sparse/0/，"
            "请更新 pipeline 或重新运行 colmap 阶段以自动规范化目录。"
        )
    return stats
