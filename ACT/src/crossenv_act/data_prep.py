from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .constants import DEFAULT_REPO_ID, DEFAULT_REVISION, ENV_NAMES
from .paths import raw_root, prepared_root, prepared_split_root, prepared_joint_root, resolve_path
from .schema_fix import standardize_act_schema

V21 = {"v2.1", "2.1"}
V3_PREFIX = ("v3", "3")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _info_path(root: Path) -> Path:
    return root / "meta" / "info.json"


def is_lerobot_root(root: Path) -> bool:
    return _info_path(root).is_file()


def codebase_version(root: Path) -> str:
    return str(_read_json(_info_path(root)).get("codebase_version") or "unknown")


def split_source_roots(raw: Path) -> dict[str, Path]:
    """Return exact CALVIN split roots, ignoring *_old, *_v30 and prepared outputs."""
    roots: dict[str, Path] = {}
    for env in ENV_NAMES:
        candidates = [raw / f"split{env}", raw / f"split{env.lower()}", raw / env, raw / env.lower()]
        for c in candidates:
            if is_lerobot_root(c):
                roots[env] = c
                break
    if roots:
        missing = [e for e in ENV_NAMES if e not in roots]
        if missing:
            raise RuntimeError(f"Found partial split dataset in {raw}; missing split(s): {missing}")
        return roots
    if is_lerobot_root(raw):
        raise RuntimeError(
            f"{raw} is a single LeRobot root, but this homework expects splitA/splitB/splitC/splitD. "
            "Please check that the correct dataset xiaoma26/calvin-lerobot was downloaded."
        )
    infos = sorted(raw.glob("*/meta/info.json"))
    raise RuntimeError(
        f"Could not locate splitA/splitB/splitC/splitD under {raw}.\n"
        f"Found meta/info.json files: {[str(p) for p in infos[:20]]}"
    )


def download_raw_dataset(
    data_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    force_download: bool = False,
) -> Path:
    raw = raw_root(data_dir)
    if force_download and raw.exists():
        shutil.rmtree(raw)
    if raw.exists():
        try:
            split_source_roots(raw)
            tqdm.write(f"[data] Reusing raw dataset at {raw}")
            return raw
        except Exception:
            if not force_download:
                tqdm.write(f"[data] Existing raw directory is incomplete; re-downloading into {raw}")
                shutil.rmtree(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"[data] Downloading {repo_id} to {raw}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=str(raw))
    except Exception as exc:
        tqdm.write(f"[data] snapshot_download failed ({exc}); falling back to `hf download`.")
        cmd = ["hf", "download", repo_id, "--repo-type=dataset", "--revision", revision, "--local-dir", str(raw)]
        subprocess.run(cmd, check=True)
    split_source_roots(raw)
    return raw


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        return
    # Metadata files are copied because we may patch them. Large payload files are hardlinked.
    if "data" in src.parts or "videos" in src.parts or "images" in src.parts:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def materialize_split_to_work(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            _copy_or_link(p, q)


def patch_v21_count_fields(root: Path) -> int:
    """Patch legacy v2.1 episode stats whose per-feature stats miss `count`.

    Official LeRobot conversion aggregates stats using mean/std/count. Some CALVIN
    v2.1 stats omit count for one or more features. We infer count from the legacy
    per-episode length, which is exactly what count represents for frame-wise stats.
    """
    ep_path = root / "meta" / "episodes.jsonl"
    st_path = root / "meta" / "episodes_stats.jsonl"
    if not ep_path.is_file() or not st_path.is_file():
        return 0
    lengths: dict[int, int] = {}
    with ep_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row.get("length", 1))
    changed = 0
    rows: list[dict[str, Any]] = []
    with st_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ep = int(row.get("episode_index", len(rows)))
            count = [int(lengths.get(ep, 1))]
            stats = row.get("stats", {})
            if isinstance(stats, dict):
                for _, ft_stats in stats.items():
                    if isinstance(ft_stats, dict) and "count" not in ft_stats:
                        ft_stats["count"] = count
                        changed += 1
            rows.append(row)
    if changed:
        backup = st_path.with_suffix(st_path.suffix + ".countfix.bak")
        if not backup.exists():
            shutil.copy2(st_path, backup)
        with st_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tqdm.write(f"[convert] Added missing count fields in {st_path}: {changed}")
    return changed


def normalize_v21_marker(root: Path) -> None:
    info = _read_json(_info_path(root))
    if str(info.get("codebase_version")) == "2.1":
        info["codebase_version"] = "v2.1"
        _write_json(_info_path(root), info)


def run_official_v21_to_v30(root: Path, repo_id: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        f"--root={root}",
        "--push-to-hub=false",
    ]
    tqdm.write("[convert] " + " ".join(map(str, cmd)))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"LeRobot v2.1 -> v3.0 conversion failed with exit code {proc.returncode}. Command: {' '.join(map(str, cmd))}")
    old = root.parent / f"{root.name}_old"
    if old.exists():
        shutil.rmtree(old)


def _validate_prepared_root(root: Path) -> None:
    info = _read_json(_info_path(root))
    features = info.get("features", {}) or {}
    if "action" not in features:
        raise RuntimeError(f"Prepared root {root} has no standard 'action' feature. Features={sorted(features)}")
    if not any(str(ft.get("dtype", "")).lower() in {"image", "video"} for ft in features.values() if isinstance(ft, dict)) \
       and "observation.environment_state" not in features:
        raise RuntimeError(f"Prepared root {root} has no visual/environment input for ACT. Features={sorted(features)}")
    version = str(info.get("codebase_version", ""))
    if not (version.startswith("v3") or version.startswith("3")):
        raise RuntimeError(f"Prepared root {root} is not v3.x after conversion: codebase_version={version}")


def prepare_one_split(
    env: str,
    src: Path,
    data_dir: str | Path,
    repo_id: str,
    *,
    force_prepare: bool = False,
) -> Path:
    env = env.upper()
    out = prepared_split_root(data_dir, env)
    report = out / "meta" / "act_schema_report.json"
    if out.exists() and report.is_file() and not force_prepare:
        try:
            _validate_prepared_root(out)
            tqdm.write(f"[prepare] Reusing prepared split{env}: {out}")
            return out
        except Exception as exc:
            tqdm.write(f"[prepare] Existing prepared split{env} invalid ({exc}); rebuilding.")
    prep = prepared_root(data_dir)
    work = prep / ".work" / f"split{env}"
    if work.exists():
        shutil.rmtree(work)
    if out.exists():
        shutil.rmtree(out)
    materialize_split_to_work(src, work)
    version = codebase_version(work)
    tqdm.write(f"[prepare] split{env}: source version={version}")
    if version in V21:
        patch_v21_count_fields(work)
        normalize_v21_marker(work)
        run_official_v21_to_v30(work, repo_id)
    elif version.startswith(V3_PREFIX):
        pass
    else:
        raise RuntimeError(f"Unsupported LeRobot codebase_version for split{env}: {version}")
    standardize_act_schema(work)
    _validate_prepared_root(work)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(work), str(out))
    # Remove any converter/work residues under .work.
    work_parent = prep / ".work"
    for p in list(work_parent.glob(f"split{env}_old*")) + list(work_parent.glob(f"split{env}_v30*")):
        if p.exists():
            shutil.rmtree(p)
    tqdm.write(f"[prepare] Prepared split{env}: {out}")
    return out


def prepare_splits(
    data_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    force_download: bool = False,
    force_prepare: bool = False,
) -> dict[str, Path]:
    raw = download_raw_dataset(data_dir, repo_id, revision, force_download=force_download)
    sources = split_source_roots(raw)
    prep = prepared_root(data_dir)
    prep.mkdir(parents=True, exist_ok=True)
    env_roots = {env: prepare_one_split(env, sources[env], data_dir, repo_id, force_prepare=force_prepare) for env in ENV_NAMES}
    return env_roots


def _read_info(root: Path) -> dict[str, Any]:
    return _read_json(root / "meta" / "info.json")


def _write_info(root: Path, info: dict[str, Any]) -> None:
    _write_json(root / "meta" / "info.json", info)


def _episode_tables(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("**/*.parquet"))


def _read_episode_table(root: Path) -> pd.DataFrame:
    paths = _episode_tables(root)
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _copy_or_link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        try:
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)


def _merge_stats_fallback(roots: list[Path], out_root: Path) -> None:
    # Try exact LeRobot aggregation first; fall back to split A stats because all splits share schema.
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.io_utils import load_stats, write_stats
        stats_list = [load_stats(r) for r in roots]
        write_stats(aggregate_stats(stats_list), out_root)
        return
    except Exception as exc:
        tqdm.write(f"[merge warning] Could not aggregate stats exactly ({exc}); using stats from first split.")
    shutil.copy2(roots[0] / "meta" / "stats.json", out_root / "meta" / "stats.json")


def merge_abc(data_dir: str | Path, *, force_prepare: bool = False) -> Path:
    roots = [prepared_split_root(data_dir, e) for e in ("A", "B", "C")]
    for r in roots:
        _validate_prepared_root(r)
    out = prepared_joint_root(data_dir)
    signature = {"roots": [str(r.resolve()) for r in roots], "version": 2}
    marker = out / "meta" / ".merge_signature.json"
    if out.exists() and marker.is_file() and not force_prepare:
        try:
            if _read_json(marker) == signature:
                _validate_prepared_root(out)
                tqdm.write(f"[merge] Reusing joint A/B/C dataset: {out}")
                return out
        except Exception:
            pass
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    base_info = _read_info(roots[0])
    all_rows: list[dict[str, Any]] = []
    ep_offset = 0
    frame_offset = 0
    data_global = 0
    video_global = 0
    for root in roots:
        info = _read_info(root)
        if set(info.get("features", {})) != set(base_info.get("features", {})):
            raise RuntimeError(f"Feature keys differ between {roots[0]} and {root}; cannot merge safely.")
        ep_df = _read_episode_table(root)
        data_map: dict[tuple[int, int], tuple[int, int]] = {}
        video_map: dict[tuple[str, int, int], tuple[int, int]] = {}
        for src_data in sorted((root / "data").glob("chunk-*/file-*.parquet")):
            rel = src_data.relative_to(root / "data")
            src_chunk = int(rel.parts[0].split("-")[-1])
            src_file = int(rel.parts[1].split("-")[-1].split(".")[0])
            dst_chunk, dst_file = data_global, 0
            data_global += 1
            data_map[(src_chunk, src_file)] = (dst_chunk, dst_file)
            dst = out / "data" / f"chunk-{dst_chunk:03d}" / f"file-{dst_file:03d}.parquet"
            df = pd.read_parquet(src_data)
            if "episode_index" in df.columns:
                df["episode_index"] = df["episode_index"].astype("int64") + ep_offset
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
        for src_video in sorted((root / "videos").glob("*/*/file-*.mp4")):
            rel = src_video.relative_to(root / "videos")
            video_key = rel.parts[0]
            src_chunk = int(rel.parts[1].split("-")[-1])
            src_file = int(rel.parts[2].split("-")[-1].split(".")[0])
            dst_chunk, dst_file = video_global, 0
            video_global += 1
            video_map[(video_key, src_chunk, src_file)] = (dst_chunk, dst_file)
            dst = out / "videos" / video_key / f"chunk-{dst_chunk:03d}" / f"file-{dst_file:03d}.mp4"
            _copy_or_link_file(src_video, dst)
        for _, row in ep_df.iterrows():
            d = row.to_dict()
            if "episode_index" in d:
                d["episode_index"] = int(d["episode_index"]) + ep_offset
            if "dataset_from_index" in d:
                d["dataset_from_index"] = int(d["dataset_from_index"]) + frame_offset
            if "dataset_to_index" in d:
                d["dataset_to_index"] = int(d["dataset_to_index"]) + frame_offset
            if "data/chunk_index" in d and "data/file_index" in d:
                old = (int(d["data/chunk_index"]), int(d["data/file_index"]))
                if old in data_map:
                    d["data/chunk_index"], d["data/file_index"] = data_map[old]
            for key in list(d):
                if key.startswith("videos/") and key.endswith("/chunk_index"):
                    vk = key[len("videos/"):-len("/chunk_index")]
                    fk = f"videos/{vk}/file_index"
                    if fk in d:
                        old = (vk, int(d[key]), int(d[fk]))
                        if old in video_map:
                            d[key], d[fk] = video_map[old]
            d["meta/episodes/chunk_index"] = 0
            d["meta/episodes/file_index"] = 0
            all_rows.append(d)
        ep_offset += int(info.get("total_episodes", len(ep_df)))
        frame_offset += int(info.get("total_frames", ep_df.get("length", pd.Series([0])).sum()))
    merged_info = dict(base_info)
    merged_info["total_episodes"] = ep_offset
    merged_info["total_frames"] = frame_offset
    _write_info(out, merged_info)
    shutil.copy2(roots[0] / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")
    pd.DataFrame(all_rows).to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    _merge_stats_fallback(roots, out)
    standardize_act_schema(out)
    _write_json(marker, signature)
    _validate_prepared_root(out)
    tqdm.write(f"[merge] Created joint A/B/C dataset: {out}")
    return out


def prepare_all(
    data_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    *,
    force_download: bool = False,
    force_prepare: bool = False,
) -> dict[str, Path]:
    env_roots = prepare_splits(data_dir, repo_id, revision, force_download=force_download, force_prepare=force_prepare)
    env_roots["ABC"] = merge_abc(data_dir, force_prepare=force_prepare)
    return env_roots


def root_for_mode(data_dir: str | Path, mode: str) -> Path:
    mode = mode.upper().replace(",", "")
    if mode == "B":
        return prepared_split_root(data_dir, "B")
    if mode in {"ABC", "A+B+C"}:
        return prepared_joint_root(data_dir)
    if mode == "D":
        return prepared_split_root(data_dir, "D")
    if mode in ENV_NAMES:
        return prepared_split_root(data_dir, mode)
    raise ValueError(f"Unknown mode/env: {mode}")
