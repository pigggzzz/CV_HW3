from __future__ import annotations

from typing import Any, Mapping


def flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def render_template(s: str, mapping: Mapping[str, Any]) -> str:
    """
    用 str.format 渲染命令模板。
    - 推荐在 YAML 里写 `{colmap_bin}` / `{database_path}` 这样的占位符
    - 这里会同时提供“扁平键”和“原始顶层键”
    """
    flat = flatten_dict(mapping)
    merged: dict[str, Any] = dict(mapping)
    merged.update(flat)
    return s.format(**merged)

