from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


ACTION_KEY = "action"
OBS_STATE_KEY = "observation.state"
OBS_IMAGES_PREFIX = "observation.images."


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_info(root: Path) -> dict[str, Any]:
    info = _load_json(root / "meta" / "info.json")
    if not isinstance(info, dict):
        raise FileNotFoundError(f"{root / 'meta' / 'info.json'} is missing or invalid")
    return info


def _write_info(root: Path, info: dict[str, Any]) -> None:
    _write_json(root / "meta" / "info.json", info)


def _read_stats(root: Path) -> dict[str, Any]:
    stats = _load_json(root / "meta" / "stats.json", {})
    return stats if isinstance(stats, dict) else {}


def _write_stats(root: Path, stats: dict[str, Any]) -> None:
    _write_json(root / "meta" / "stats.json", stats)


def _data_parquets(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def _episode_parquets(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("**/*.parquet"))


def _normal_feature_name(name: str) -> str:
    return name.lower().replace(".", "_").replace("-", "_").replace("/", "_")


def _is_numeric_vector_feature(ft: Any) -> bool:
    if not isinstance(ft, dict):
        return False
    dtype = str(ft.get("dtype", "")).lower()
    if dtype in {"image", "video", "string"}:
        return False
    shape = ft.get("shape")
    return isinstance(shape, (list, tuple)) and len(shape) == 1


def _shape_len(ft: dict[str, Any], fallback: int = 1) -> int:
    shape = ft.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        try:
            return int(shape[0])
        except Exception:
            return fallback
    return fallback


def _zeros(shape_len: int) -> list[float]:
    return [0.0 for _ in range(max(1, int(shape_len)))]


def _ones(shape_len: int) -> list[float]:
    return [1.0 for _ in range(max(1, int(shape_len)))]


def _vector_stats_template(shape_len: int) -> dict[str, Any]:
    return {
        "min": _zeros(shape_len),
        "max": _ones(shape_len),
        "mean": _zeros(shape_len),
        "std": _ones(shape_len),
        "count": [1],
    }


def _visual_stats_template() -> dict[str, Any]:
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
        "count": [1],
    }


def _ensure_stats_entry(stats: dict[str, Any], key: str, ft: dict[str, Any]) -> bool:
    changed = False
    template = _visual_stats_template() if str(ft.get("dtype", "")).lower() in {"image", "video"} else _vector_stats_template(_shape_len(ft))
    cur = stats.get(key)
    if not isinstance(cur, dict):
        stats[key] = dict(template)
        return True
    for stat_key, value in template.items():
        if stat_key not in cur:
            cur[stat_key] = value
            changed = True
    return changed


def _rename_stats_key(stats: dict[str, Any], old: str, new: str) -> bool:
    if old == new:
        return False
    if old in stats:
        if new not in stats:
            stats[new] = stats.pop(old)
        else:
            stats.pop(old, None)
        return True
    return False


def _rename_data_columns(root: Path, rename_map: dict[str, str]) -> int:
    n_changed = 0
    for parquet in _data_parquets(root):
        df = pd.read_parquet(parquet)
        cols = {old: new for old, new in rename_map.items() if old in df.columns and new not in df.columns}
        drops = [old for old, new in rename_map.items() if old in df.columns and new in df.columns and old != new]
        if cols or drops:
            df = df.rename(columns=cols)
            if drops:
                df = df.drop(columns=drops)
            df.to_parquet(parquet, index=False)
            n_changed += 1
    return n_changed


def _rename_episode_metadata_columns(root: Path, rename_map: dict[str, str]) -> int:
    """Rename stats/<key>/... and videos/<key>/... columns in v3 episode metadata."""
    n_changed = 0
    for parquet in _episode_parquets(root):
        df = pd.read_parquet(parquet)
        col_map: dict[str, str] = {}
        for col in list(df.columns):
            for old, new in rename_map.items():
                if col.startswith(f"stats/{old}/"):
                    col_map[col] = f"stats/{new}/" + col[len(f"stats/{old}/"):]
                elif col.startswith(f"videos/{old}/"):
                    col_map[col] = f"videos/{new}/" + col[len(f"videos/{old}/"):]
        col_map = {k: v for k, v in col_map.items() if k != v and v not in df.columns}
        if col_map:
            df = df.rename(columns=col_map)
            df.to_parquet(parquet, index=False)
            n_changed += 1
    return n_changed


def _rename_video_dir(root: Path, old: str, new: str) -> bool:
    old_dir = root / "videos" / old
    new_dir = root / "videos" / new
    if old == new or not old_dir.exists():
        return False
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    if new_dir.exists():
        # Keep the existing destination and remove duplicate source to avoid ambiguity.
        shutil.rmtree(old_dir)
    else:
        shutil.move(str(old_dir), str(new_dir))
    return True


def _first_data_columns(root: Path) -> list[str]:
    for parquet in _data_parquets(root):
        try:
            return list(pd.read_parquet(parquet, columns=None).columns)
        except Exception:
            continue
    return []


def _infer_vector_shape_from_data(root: Path, col: str) -> int:
    for parquet in _data_parquets(root):
        try:
            df = pd.read_parquet(parquet, columns=[col])
        except Exception:
            continue
        for value in df[col].head(50):
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
    return 1


def _choose_action_candidate(features: dict[str, Any], data_columns: list[str]) -> str | None:
    if ACTION_KEY in features:
        return ACTION_KEY
    priority = ["actions", "rel_actions", "relative_actions", "action_delta", "action_rel", "delta_actions"]
    feature_keys = list(features)
    for p in priority:
        for k in feature_keys:
            if _normal_feature_name(k) == p and _is_numeric_vector_feature(features[k]):
                return k
    for k in feature_keys:
        nk = _normal_feature_name(k)
        if "action" in nk and _is_numeric_vector_feature(features[k]):
            return k
    for p in priority:
        for k in data_columns:
            if _normal_feature_name(k) == p:
                return k
    for k in data_columns:
        if "action" in _normal_feature_name(k):
            return k
    return None


def _choose_state_candidate(features: dict[str, Any]) -> str | None:
    if OBS_STATE_KEY in features:
        return OBS_STATE_KEY
    priority = ["observation_state", "state", "robot_obs", "robot_state", "proprio", "proprioception"]
    for p in priority:
        for k, ft in features.items():
            if _normal_feature_name(k) == p and _is_numeric_vector_feature(ft):
                return k
    return None


def _visual_target_name(old: str, used: set[str]) -> str:
    if old.startswith(OBS_IMAGES_PREFIX):
        return old
    base = old
    for prefix in ["observation.images.", "observation.image.", "observation.", "images.", "image."]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    base = base.strip("._-/") or "image"
    # Avoid slashes because LeRobot feature names should not contain '/'.
    base = base.replace("/", ".")
    target = OBS_IMAGES_PREFIX + base
    i = 1
    while target in used and target != old:
        i += 1
        target = OBS_IMAGES_PREFIX + f"{base}_{i}"
    return target


def standardize_act_schema(root: str | Path, *, write_report: bool = True) -> dict[str, Any]:
    """Normalize a converted CALVIN-LeRobot root so LeRobot ACT can infer features.

    Required by ACT/LeRobot v3:
    - the action output feature must be named exactly ``action``;
    - visual inputs should be valid visual features, preferably ``observation.images.*``;
    - stats must contain all policy feature keys.
    """
    root = Path(root).expanduser().resolve()
    info = _read_info(root)
    features = info.setdefault("features", {})
    if not isinstance(features, dict):
        raise RuntimeError(f"Invalid features in {root / 'meta' / 'info.json'}")

    stats = _read_stats(root)
    data_columns = _first_data_columns(root)
    rename_map: dict[str, str] = {}
    warnings: list[str] = []
    actions: list[str] = []

    # 1) Canonicalize action key.
    action_candidate = _choose_action_candidate(features, data_columns)
    if action_candidate is None:
        raise RuntimeError(
            f"Could not find an action feature in {root}.\n"
            f"Feature keys: {sorted(features)}\nData columns: {data_columns}\n"
            "LeRobot ACT requires a numeric output feature named exactly 'action'."
        )
    if action_candidate != ACTION_KEY:
        if action_candidate in features:
            features[ACTION_KEY] = features.pop(action_candidate)
        else:
            shape_len = _infer_vector_shape_from_data(root, action_candidate)
            features[ACTION_KEY] = {"dtype": "float32", "shape": [shape_len], "names": [f"action_{i}" for i in range(shape_len)]}
        rename_map[action_candidate] = ACTION_KEY
        _rename_stats_key(stats, action_candidate, ACTION_KEY)
        actions.append(f"renamed action feature {action_candidate!r} -> 'action'")
    _ensure_stats_entry(stats, ACTION_KEY, features[ACTION_KEY])

    # 2) Canonicalize robot state when obvious. This is optional for ACT but useful.
    state_candidate = _choose_state_candidate(features)
    if state_candidate and state_candidate != OBS_STATE_KEY:
        features[OBS_STATE_KEY] = features.pop(state_candidate)
        rename_map[state_candidate] = OBS_STATE_KEY
        _rename_stats_key(stats, state_candidate, OBS_STATE_KEY)
        actions.append(f"renamed state feature {state_candidate!r} -> 'observation.state'")
    if OBS_STATE_KEY in features:
        _ensure_stats_entry(stats, OBS_STATE_KEY, features[OBS_STATE_KEY])

    # 3) Canonicalize visual keys to observation.images.* and guarantee stats.
    used = set(features)
    visual_renames: dict[str, str] = {}
    for key, ft in list(features.items()):
        if isinstance(ft, dict) and str(ft.get("dtype", "")).lower() in {"image", "video"}:
            target = _visual_target_name(key, used)
            if target != key:
                used.add(target)
                visual_renames[key] = target
                features[target] = features.pop(key)
                _rename_stats_key(stats, key, target)
                rename_map[key] = target
                _rename_video_dir(root, key, target)
                actions.append(f"renamed visual feature {key!r} -> {target!r}")
    for key, ft in features.items():
        if isinstance(ft, dict) and str(ft.get("dtype", "")).lower() in {"image", "video"}:
            _ensure_stats_entry(stats, key, ft)

    # 4) Apply physical parquet metadata/data column renames.
    data_changed = _rename_data_columns(root, rename_map) if rename_map else 0
    episodes_changed = _rename_episode_metadata_columns(root, rename_map) if rename_map else 0

    # 5) Save metadata.
    _write_info(root, info)
    _write_stats(root, stats)

    # 6) Validate locally instead of importing LeRobot private helpers.
    # Different LeRobot releases move/remove `lerobot.utils.feature_utils`; relying on
    # that private path produced noisy warnings on some installations. The following
    # mirrors the conditions ACT needs: a numeric output key exactly named `action`,
    # plus at least one visual input or environment state.
    visual_keys = [
        key for key, ft in features.items()
        if isinstance(ft, dict) and str(ft.get("dtype", "")).lower() in {"image", "video"}
    ]
    state_keys = [
        key for key, ft in features.items()
        if isinstance(ft, dict) and key.startswith("observation") and str(ft.get("dtype", "")).lower() not in {"image", "video", "string"}
    ]
    action_ok = ACTION_KEY in features and _is_numeric_vector_feature(features[ACTION_KEY])
    validation: dict[str, Any] = {
        "has_action": bool(action_ok),
        "visual_keys": sorted(visual_keys),
        "state_keys": sorted(state_keys),
        "policy_feature_keys": sorted([ACTION_KEY] + visual_keys + state_keys) if action_ok else sorted(visual_keys + state_keys),
    }
    if not action_ok:
        raise RuntimeError(
            f"After schema fix, feature 'action' is still missing or not a numeric vector. Features: {features}"
        )
    if not visual_keys and "observation.environment_state" not in features:
        warnings.append("No visual feature was detected; ACT will fail unless environment_state is present.")

    report = {
        "root": str(root),
        "actions": actions,
        "warnings": warnings,
        "rename_map": rename_map,
        "data_parquet_files_rewritten": data_changed,
        "episode_metadata_files_rewritten": episodes_changed,
        "features": sorted(features.keys()),
        "stats_keys": sorted(stats.keys()),
        "validation": validation,
    }
    if write_report:
        report_path = root / "meta" / "act_schema_report.json"
        _write_json(report_path, report)
        if actions or warnings:
            tqdm.write(f"[schema] Wrote ACT schema report: {report_path}")
            for a in actions:
                tqdm.write(f"[schema] {a}")
            for w in warnings:
                tqdm.write(f"[schema warning] {w}")
    return report
