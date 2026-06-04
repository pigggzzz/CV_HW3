from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .paths import ensure_dir


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def write_json(path: str | Path, obj: Mapping[str, Any]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p

