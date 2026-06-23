from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm.auto import tqdm

from .constants import DEFAULT_REPO_ID, DEFAULT_REVISION
from .data_prep import prepare_all, root_for_mode
from .eval import load_policy, make_lerobot_dataset, make_preprocessor, move_to_device, predict_chunk
from .paths import resolve_path
from .train import pretrained_dir, runtime_env, write_json

ACTION = "action"


class ReplayVisualizationError(RuntimeError):
    pass


@dataclass
class ReplayConfig:
    data_dir: Path
    output_dir: Path
    repo_id: str
    revision: str
    cuda_id: str
    basic_policy: Path
    joint_policy: Path
    num_sequences: int
    sequence_ids: list[int] | None
    start_indices: list[int] | None
    list_candidates_only: bool
    fast_generate: bool
    steps_per_sequence: int
    candidate_stride: int
    min_frame_index: int
    end_margin: int
    min_valid_horizon: int
    fps: int
    tolerance_s: float
    force_download: bool
    force_prepare: bool
    wandb_enable: bool
    wandb_project: str
    wandb_mode: str


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).replace(";", ",").replace(" ", ",").split(",")]
    vals = [int(p) for p in parts if p != ""]
    return vals or None


def _scalar_int(value: Any) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    try:
        return int(arr.reshape(-1)[0].item())
    except Exception:
        try:
            return int(value)
        except Exception:
            return None


def _as_numpy_1d(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _as_numpy_image(x: Any) -> np.ndarray:
    """Convert a LeRobot image tensor/array to uint8 HWC for display."""
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # Strip batch if present.
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:  # CHW -> HWC
        arr = np.transpose(arr[:3], (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ReplayVisualizationError(f"Cannot convert image with shape {arr.shape} to HWC image.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _find_image_key(sample: dict[str, Any], dataset: Any) -> str:
    meta = getattr(dataset, "meta", None)
    camera_keys = list(getattr(meta, "camera_keys", []) or [])
    for key in camera_keys:
        if key in sample:
            return key
    candidates = [k for k in sample.keys() if isinstance(k, str) and (k.startswith("observation.images") or k.startswith("observation.image"))]
    if candidates:
        for pref in ["observation.images.image", "observation.image"]:
            if pref in candidates:
                return pref
        return sorted(candidates)[0]
    raise ReplayVisualizationError("No image observation key found. Expected a key such as 'observation.images.image'.")


def _convert_camera_uint8_to_float(batch: dict[str, Any], dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    camera_keys = list(getattr(meta, "camera_keys", []) or [])
    for cam_key in camera_keys:
        if cam_key in batch and torch.is_tensor(batch[cam_key]) and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    for key in list(batch.keys()):
        if isinstance(key, str) and key.startswith("observation.images"):
            if torch.is_tensor(batch[key]) and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    return batch


def _sample_episode_index(sample: dict[str, Any]) -> int | None:
    for key in ("episode_index", "episode_idx", "episode_id"):
        if key in sample:
            return _scalar_int(sample[key])
    return None


def _sample_frame_index(sample: dict[str, Any]) -> int | None:
    for key in ("frame_index", "frame_idx", "index"):
        if key in sample:
            return _scalar_int(sample[key])
    return None


def _sample_valid_horizon(sample: dict[str, Any]) -> int | None:
    if "action_is_pad" in sample:
        pad = _as_numpy_1d(sample["action_is_pad"]).astype(bool)
        return int((~pad).sum())
    if ACTION in sample:
        arr = np.asarray(sample[ACTION])
        if arr.ndim >= 2:
            return int(arr.shape[0])
    return None


def _episode_ranges_from_dataset(dataset: Any) -> list[dict[str, int]]:
    """Return episode global index ranges as half-open [start, end) intervals.

    LeRobotDataset commonly exposes `episode_data_index` with `from` and `to`.
    We use it to avoid sampling visualization windows that cross episode boundaries.
    """
    edi = getattr(dataset, "episode_data_index", None)
    if edi is None and hasattr(dataset, "dataset"):
        edi = getattr(dataset.dataset, "episode_data_index", None)
    if edi is None:
        return []

    start_obj = None
    end_obj = None
    for sk in ("from", "start", "starts"):
        if isinstance(edi, dict) and sk in edi:
            start_obj = edi[sk]
            break
    for ek in ("to", "end", "ends"):
        if isinstance(edi, dict) and ek in edi:
            end_obj = edi[ek]
            break
    if start_obj is None or end_obj is None:
        return []

    starts = _as_numpy_1d(start_obj).astype(int)
    ends = _as_numpy_1d(end_obj).astype(int)
    n = min(len(starts), len(ends))
    out: list[dict[str, int]] = []
    for ep in range(n):
        s = int(starts[ep])
        e = int(ends[ep])
        if e > s:
            out.append({"episode_index": ep, "start_index": s, "end_index": e, "episode_length": e - s})
    return out


def _fallback_episode_range_by_sampling(dataset: Any, idx: int) -> dict[str, int] | None:
    """Infer an episode range around idx using sample episode_index.

    This fallback is slower and only used when LeRobot metadata is unavailable.
    """
    sample = dataset[idx]
    if not isinstance(sample, dict):
        return None
    ep = _sample_episode_index(sample)
    if ep is None:
        return None
    left = idx
    while left > 0:
        prev = dataset[left - 1]
        if not isinstance(prev, dict) or _sample_episode_index(prev) != ep:
            break
        left -= 1
    right = idx + 1
    n = len(dataset)
    while right < n:
        nxt = dataset[right]
        if not isinstance(nxt, dict) or _sample_episode_index(nxt) != ep:
            break
        right += 1
    return {"episode_index": ep, "start_index": left, "end_index": right, "episode_length": right - left}


def _valid_action_tensors(pred: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ACTION not in batch:
        raise ReplayVisualizationError("Batch does not contain 'action'; prepared splitD schema is invalid.")
    target = batch[ACTION]
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise ReplayVisualizationError("Predicted action and target action must both be tensors.")
    if pred.ndim != 3 or target.ndim != 3:
        raise ReplayVisualizationError(f"Expected [B,T,D] actions, got pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    t = min(pred.shape[1], target.shape[1])
    d = min(pred.shape[2], target.shape[2])
    if t <= 0 or d <= 0:
        raise ReplayVisualizationError(f"Invalid action shapes: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    pred = pred[:, :t, :d]
    target = target[:, :t, :d]
    if "action_is_pad" in batch and torch.is_tensor(batch["action_is_pad"]):
        mask = (~batch["action_is_pad"][:, :t]).detach().bool()
    else:
        mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    if mask.shape != target.shape[:2]:
        raise ReplayVisualizationError(f"action_is_pad shape mismatch: mask={tuple(mask.shape)}, action={tuple(target.shape)}")
    if not torch.isfinite(pred).all():
        raise ReplayVisualizationError("Predicted action contains NaN or Inf.")
    if not torch.isfinite(target).all():
        raise ReplayVisualizationError("Target action contains NaN or Inf.")
    return pred, target, mask


def _valid_len_from_mask(mask: torch.Tensor) -> int:
    valid = mask[0].detach().cpu().numpy().astype(bool)
    return int(valid.sum())


def _prefix_valid_len_from_mask(mask_1d: np.ndarray) -> int:
    """Return the valid prefix length from a LeRobot action_is_pad mask.

    Action chunks should have a valid prefix followed by padding near episode ends.
    For visualization, using the prefix is safer than keeping scattered valid
    positions, because the x-axis should remain the original future horizon.
    """
    valid = np.asarray(mask_1d).astype(bool).reshape(-1)
    invalid = np.flatnonzero(~valid)
    if len(invalid) == 0:
        return int(len(valid))
    return int(invalid[0])


def _clip_to_valid_horizon(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    episode_remaining: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Clip action chunks to horizons that are meaningful for this frame.

    This fixes the earlier visualization issue: the displayed chunk must not
    include padding or future actions beyond the current episode.  We do not
    search for only "safe" videos anymore; instead, every selected frame is
    clipped to its valid prefix before plotting and before computing mean L1.
    """
    valid_mask = mask[0].detach().cpu().numpy().astype(bool)
    valid_len = _prefix_valid_len_from_mask(valid_mask)
    if episode_remaining is not None:
        valid_len = min(valid_len, max(0, int(episode_remaining)))
    valid_len = min(valid_len, int(pred.shape[1]), int(target.shape[1]))
    if valid_len <= 0:
        raise ReplayVisualizationError(
            "No valid action horizon remains after clipping padding and episode boundary. "
            "Choose a start frame farther from the episode end."
        )
    was_clipped = valid_len < int(pred.shape[1])
    return pred[:, :valid_len, :], target[:, :valid_len, :], int(valid_len), bool(was_clipped)


def _metrics_from_action_chunk(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = (pred - target).abs()[0]  # [T,D], already clipped to valid horizons
    err_np = err.detach().cpu().numpy()
    horizon_l1 = err_np.mean(axis=-1)
    return {
        "chunk_l1": float(horizon_l1.mean()),
        "first_action_l1": float(horizon_l1[0]),
        "tail_action_l1": float(horizon_l1[-max(1, min(10, len(horizon_l1))):].mean()),
        "valid_horizon": int(len(horizon_l1)),
    }


def _make_panel_frame(
    image: np.ndarray,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    local_step: int,
    dataset_index: int,
    episode_index: int | None,
    frame_index: int | None,
    title: str,
) -> np.ndarray:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ReplayVisualizationError("matplotlib is required for replay visualization. Install it with `pip install matplotlib`.") from exc

    pred_np = pred[0].detach().cpu().numpy()
    target_np = target[0].detach().cpu().numpy()
    err_h = np.abs(pred_np - target_np).mean(axis=-1)
    h = np.arange(len(err_h))
    pred_profile = pred_np.mean(axis=-1)
    target_profile = target_np.mean(axis=-1)

    fig = plt.figure(figsize=(11, 4.2), dpi=110)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(image)
    epi = "?" if episode_index is None else str(episode_index)
    fri = "?" if frame_index is None else str(frame_index)
    ax0.set_title(f"splitD observation\nep {epi}, frame {fri}\nidx {dataset_index}, step {local_step}")
    ax0.axis("off")

    ax1.plot(h, target_profile, label="expert")
    ax1.plot(h, pred_profile, label="pred")
    ax1.set_title(f"Valid action chunk profile\nvalid horizon={len(h)}")
    ax1.set_xlabel("horizon")
    ax1.set_ylabel("mean action value")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2.plot(h, err_h)
    ax2.set_title(f"Valid-horizon L1 error\nmean={err_h.mean():.3f}")
    ax2.set_xlabel("horizon")
    ax2.set_ylabel("L1 error")
    ax2.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ReplayVisualizationError("No frames were generated; cannot save video.")
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise ReplayVisualizationError("imageio is required to save mp4 videos. Install it with `pip install imageio imageio-ffmpeg`.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=16)
    except Exception as exc:
        raise ReplayVisualizationError(
            f"Failed to write {path}. If ffmpeg is missing, install `imageio-ffmpeg` or save as GIF instead. Original error: {exc}"
        ) from exc


def _load_dataset_for_replay(cfg: ReplayConfig):
    device = torch.device("cpu" if str(cfg.cuda_id).lower() in {"cpu", "none", "-1"} or not torch.cuda.is_available() else "cuda")
    basic_policy, _ = load_policy(cfg.basic_policy, device)
    dataset_root = root_for_mode(cfg.data_dir, "D")
    ds_out = make_lerobot_dataset(cfg.repo_id, dataset_root, cfg.revision, basic_policy.config, cfg.tolerance_s)
    dataset, ds_meta = ds_out[0], ds_out[1]
    return dataset, ds_meta, device


def _build_replay_windows(dataset: Any, cfg: ReplayConfig) -> list[dict[str, int]]:
    """Build replay windows without scanning the whole dataset.

    Previous versions tried to enumerate "safe candidates" by scanning many
    possible windows.  That is slow on large splitD.  The robust fix is to
    allow arbitrary selected episodes/windows and clip each action chunk to the
    valid horizon at rendering time.  Here we only choose start locations from
    episode metadata when available; no exhaustive safety scan is performed.
    """
    n = len(dataset)
    if n <= 0:
        raise ReplayVisualizationError("The splitD dataset is empty.")
    if cfg.sequence_ids is not None and cfg.start_indices is not None:
        raise ReplayVisualizationError("Use either --sequence-ids or --start-indices, not both.")

    ranges = _episode_ranges_from_dataset(dataset)
    windows: list[dict[str, int]] = []

    if cfg.start_indices is not None:
        # Exact dataset starts.  We infer the containing episode when possible
        # and later clip frames/horizons to avoid boundary leakage.
        for seq_id, start_idx in enumerate(cfg.start_indices):
            if start_idx < 0 or start_idx >= n:
                raise ReplayVisualizationError(f"Requested start index {start_idx} is out of dataset range [0,{n}).")
            ep_range = None
            for ep in ranges:
                if ep["start_index"] <= start_idx < ep["end_index"]:
                    ep_range = ep
                    break
            if ep_range is None:
                ep_range = _fallback_episode_range_by_sampling(dataset, start_idx)
            ep_idx = int(ep_range["episode_index"]) if ep_range else -1
            ep_start = int(ep_range["start_index"]) if ep_range else -1
            ep_end = int(ep_range["end_index"]) if ep_range else min(n, start_idx + cfg.steps_per_sequence)
            max_steps = max(1, min(cfg.steps_per_sequence, ep_end - start_idx))
            windows.append({
                "sequence_id": int(seq_id),
                "requested_id": int(start_idx),
                "selection_mode": "start_index",
                "start_index": int(start_idx),
                "episode_index": ep_idx,
                "frame_index": int(start_idx - ep_start) if ep_start >= 0 else int(start_idx),
                "episode_start_index": ep_start,
                "episode_end_index": ep_end,
                "steps_per_sequence": int(max_steps),
                "requested_steps_per_sequence": int(cfg.steps_per_sequence),
            })
        return windows

    if ranges:
        # Treat --sequence-ids as episode ids.  This is stable and fast, and it
        # matches the user's need to manually choose videos such as 2,10,17,21.
        if cfg.sequence_ids is not None:
            requested_eps = cfg.sequence_ids
        else:
            # Choose diverse episodes without scanning all candidate starts.
            available = [ep for ep in ranges if ep["episode_length"] > max(1, cfg.min_frame_index)]
            if not available:
                available = ranges
            count = min(max(1, int(cfg.num_sequences)), len(available))
            idxs = np.linspace(0, len(available) - 1, num=count, dtype=int).tolist()
            requested_eps = [int(available[i]["episode_index"]) for i in idxs]

        by_ep = {int(ep["episode_index"]): ep for ep in ranges}
        for out_id, ep_id in enumerate(requested_eps):
            if int(ep_id) not in by_ep:
                raise ReplayVisualizationError(
                    f"Requested episode/sequence id {ep_id} was not found. "
                    f"Valid episode ids are roughly 0..{len(ranges)-1}."
                )
            ep = by_ep[int(ep_id)]
            ep_start = int(ep["start_index"])
            ep_end = int(ep["end_index"])
            ep_len = int(ep["episode_length"])
            # Use min_frame_index as the start frame inside each selected episode;
            # if the episode is very short, fall back to the first frame.
            local_start = min(max(0, int(cfg.min_frame_index)), max(0, ep_len - 1))
            start_idx = ep_start + local_start
            max_steps = max(1, min(int(cfg.steps_per_sequence), ep_end - start_idx))
            windows.append({
                "sequence_id": int(ep_id),
                "requested_id": int(ep_id),
                "selection_mode": "episode_id",
                "start_index": int(start_idx),
                "episode_index": int(ep_id),
                "frame_index": int(local_start),
                "episode_start_index": int(ep_start),
                "episode_end_index": int(ep_end),
                "steps_per_sequence": int(max_steps),
                "requested_steps_per_sequence": int(cfg.steps_per_sequence),
            })
        return windows

    # Metadata-free fallback: use deterministic dataset starts.  This is not as
    # semantically stable as episode ids, but chunk clipping still prevents
    # padding from contaminating plotted horizon errors.
    if cfg.sequence_ids is not None:
        starts = [int(x) * max(1, int(cfg.candidate_stride)) for x in cfg.sequence_ids]
    else:
        starts = [i * max(1, int(cfg.candidate_stride)) for i in range(max(1, int(cfg.num_sequences)))]
    for out_id, start_idx in enumerate(starts):
        start_idx = min(max(0, int(start_idx)), max(0, n - 1))
        max_steps = max(1, min(int(cfg.steps_per_sequence), n - start_idx))
        windows.append({
            "sequence_id": int(out_id if cfg.sequence_ids is None else cfg.sequence_ids[out_id]),
            "requested_id": int(start_idx),
            "selection_mode": "dataset_stride_fallback",
            "start_index": int(start_idx),
            "episode_index": -1,
            "frame_index": int(start_idx),
            "episode_start_index": -1,
            "episode_end_index": -1,
            "steps_per_sequence": int(max_steps),
            "requested_steps_per_sequence": int(cfg.steps_per_sequence),
        })
    return windows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.4f}"
                else:
                    out[k] = v
            writer.writerow(out)


def _evaluate_and_visualize_model(
    *,
    model_label: str,
    policy_path: Path,
    dataset: Any,
    ds_meta: Any,
    device: torch.device,
    out_dir: Path,
    cfg: ReplayConfig,
    windows: list[dict[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    policy, pdir = load_policy(policy_path, device)
    preprocessor = make_preprocessor(policy, pdir, ds_meta.stats)
    policy.eval()

    video_dir = out_dir / "videos" / model_label
    rows: list[dict[str, Any]] = []
    seq_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for window in tqdm(windows, desc=f"Replay visualize {model_label}", unit="seq"):
            seq_id = int(window["sequence_id"])
            start = int(window["start_index"])
            ep_idx = int(window.get("episode_index", -1))
            frames: list[np.ndarray] = []
            seq_metrics: list[float] = []
            valid_horizons: list[int] = []
            actual_steps = int(window.get("steps_per_sequence", cfg.steps_per_sequence))
            ep_end_for_clip = int(window.get("episode_end_index", -1))
            for local_step in range(actual_steps):
                idx = start + local_step
                if idx >= len(dataset):
                    raise ReplayVisualizationError(f"Dataset index {idx} is out of range for selected sequence {seq_id}.")
                sample = dataset[idx]
                if not isinstance(sample, dict):
                    raise ReplayVisualizationError(f"Dataset item at index {idx} is not a dict.")
                sample_ep = _sample_episode_index(sample)
                if ep_idx >= 0 and sample_ep is not None and int(sample_ep) != ep_idx:
                    raise ReplayVisualizationError(
                        f"Selected sequence {seq_id} crosses episode boundary at dataset index {idx}: "
                        f"expected episode {ep_idx}, got {sample_ep}."
                    )
                sample_valid = _sample_valid_horizon(sample)
                image_key = _find_image_key(sample, dataset)
                image = _as_numpy_image(sample[image_key])
                frame_index = _sample_frame_index(sample)

                raw_batch = default_collate([sample])
                raw_batch = _convert_camera_uint8_to_float(raw_batch, dataset)
                batch = preprocessor(raw_batch)
                batch = move_to_device(batch, device)
                pred = predict_chunk(policy, batch)
                pred, target, mask = _valid_action_tensors(pred, batch)
                episode_remaining = (ep_end_for_clip - idx) if ep_end_for_clip > 0 else None
                pred_valid, target_valid, valid_len, was_clipped = _clip_to_valid_horizon(
                    pred, target, mask, episode_remaining=episode_remaining
                )
                m = _metrics_from_action_chunk(pred_valid, target_valid)
                seq_metrics.append(m["chunk_l1"])
                valid_horizons.append(m["valid_horizon"])
                rows.append({
                    "model": model_label,
                    "sequence_id": seq_id,
                    "dataset_index": idx,
                    "episode_index": sample_ep if sample_ep is not None else ep_idx,
                    "frame_index": frame_index if frame_index is not None else "",
                    "local_step": local_step,
                    "chunk_l1": m["chunk_l1"],
                    "first_action_l1": m["first_action_l1"],
                    "tail_action_l1": m["tail_action_l1"],
                    "valid_horizon": m["valid_horizon"],
                    "sample_valid_horizon": sample_valid if sample_valid is not None else "",
                    "was_clipped": was_clipped,
                    "below_min_valid_horizon": bool(m["valid_horizon"] < cfg.min_valid_horizon),
                })
                frames.append(
                    _make_panel_frame(
                        image,
                        pred_valid,
                        target_valid,
                        local_step=local_step,
                        dataset_index=idx,
                        episode_index=sample_ep if sample_ep is not None else ep_idx,
                        frame_index=frame_index,
                        title=f"{model_label}: offline splitD replay",
                    )
                )
            mean_l1 = float(np.mean(seq_metrics)) if seq_metrics else math.nan
            video_path = video_dir / f"seqid_{seq_id:04d}_start_{start}_mean_l1_{mean_l1:.3f}.mp4"
            _save_video(frames, video_path, cfg.fps)
            seq_summaries.append({
                "model": model_label,
                "sequence_id": seq_id,
                "start_index": start,
                "episode_index": ep_idx,
                "frame_index": int(window.get("frame_index", -1)),
                "steps": actual_steps,
                "requested_steps": cfg.steps_per_sequence,
                "mean_chunk_l1": mean_l1,
                "mean_valid_horizon": float(np.mean(valid_horizons)) if valid_horizons else math.nan,
                "min_valid_horizon_observed": int(min(valid_horizons)) if valid_horizons else 0,
                "video_path": str(video_path),
            })

    if not rows:
        raise ReplayVisualizationError(f"No replay rows generated for {model_label}.")
    mean_chunk = float(np.mean([r["chunk_l1"] for r in rows]))
    mean_first = float(np.mean([r["first_action_l1"] for r in rows]))
    mean_tail = float(np.mean([r["tail_action_l1"] for r in rows]))
    summary = {
        "model": model_label,
        "policy_dir": str(pdir),
        "num_sequences": len(windows),
        "requested_steps_per_sequence": cfg.steps_per_sequence,
        "min_valid_horizon_reference": cfg.min_valid_horizon,
        "mean_chunk_l1": mean_chunk,
        "mean_first_action_l1": mean_first,
        "mean_tail_action_l1": mean_tail,
        "video_dir": str(video_dir),
    }
    return summary, rows, seq_summaries


def replay_entry(args) -> int:
    cfg = ReplayConfig(
        data_dir=resolve_path(args.data_dir),
        output_dir=resolve_path(args.output_dir),
        repo_id=args.repo_id,
        revision=args.revision,
        cuda_id=str(args.cuda_id),
        basic_policy=resolve_path(args.basic_policy),
        joint_policy=resolve_path(args.joint_policy),
        num_sequences=int(args.num_sequences),
        sequence_ids=_parse_int_list(args.sequence_ids),
        start_indices=_parse_int_list(args.start_indices),
        list_candidates_only=bool(args.list_candidates_only),
        fast_generate=bool(getattr(args, "fast_generate", True)),
        steps_per_sequence=int(args.steps_per_sequence),
        candidate_stride=int(args.candidate_stride),
        min_frame_index=int(args.min_frame_index),
        end_margin=int(args.end_margin),
        min_valid_horizon=int(args.min_valid_horizon),
        fps=int(args.fps),
        tolerance_s=float(args.tolerance_s),
        force_download=bool(args.force_download),
        force_prepare=bool(args.force_prepare),
        wandb_enable=bool(args.wandb_enable),
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )
    env = runtime_env(cfg.data_dir, cfg.output_dir, cfg.cuda_id)
    os.environ.update(env)

    prepare_all(
        cfg.data_dir,
        repo_id=cfg.repo_id,
        revision=cfg.revision,
        force_download=cfg.force_download,
        force_prepare=cfg.force_prepare,
    )
    dataset, ds_meta, device = _load_dataset_for_replay(cfg)

    out_dir = cfg.output_dir / "offline_replay_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _build_replay_windows(dataset, cfg)
    selected_csv = out_dir / "offline_replay_selected_windows.csv"
    _write_csv(
        selected_csv,
        windows,
        fieldnames=[
            "sequence_id",
            "requested_id",
            "selection_mode",
            "start_index",
            "episode_index",
            "frame_index",
            "episode_start_index",
            "episode_end_index",
            "steps_per_sequence",
            "requested_steps_per_sequence",
        ],
    )
    print(f"[replay] Selected replay windows without full candidate scan: {selected_csv}")
    print(f"[replay] Selected sequence/episode ids: {[w['sequence_id'] for w in windows]}")
    print("[replay] Invalid/padded action horizons will be clipped per frame before plotting and metric computation.")

    if cfg.list_candidates_only:
        print("[replay] --list-candidates-only is deprecated in this version; wrote selected windows and exited.")
        return 0


    basic_summary, basic_rows, basic_seq_rows = _evaluate_and_visualize_model(
        model_label="act_env_b",
        policy_path=cfg.basic_policy,
        dataset=dataset,
        ds_meta=ds_meta,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        windows=windows,
    )
    joint_summary, joint_rows, joint_seq_rows = _evaluate_and_visualize_model(
        model_label="act_env_abc",
        policy_path=cfg.joint_policy,
        dataset=dataset,
        ds_meta=ds_meta,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        windows=windows,
    )

    summary_rows = [basic_summary, joint_summary]
    comparison = {
        "joint_minus_basic_mean_chunk_l1": joint_summary["mean_chunk_l1"] - basic_summary["mean_chunk_l1"],
        "relative_improvement_of_joint": (
            (basic_summary["mean_chunk_l1"] - joint_summary["mean_chunk_l1"]) / basic_summary["mean_chunk_l1"]
            if basic_summary["mean_chunk_l1"] != 0
            else None
        ),
    }
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "test_env": "D",
        "evaluation_type": "safe offline replay visualization on LeRobot splitD; not closed-loop simulator success rate",
        "selected_sequence_ids": [int(w["sequence_id"]) for w in windows],
        "selected_start_indices": [int(w["start_index"]) for w in windows],
        "num_sequences": len(windows),
        "requested_steps_per_sequence": cfg.steps_per_sequence,
        "min_valid_horizon_reference": cfg.min_valid_horizon,
        "basic": basic_summary,
        "joint": joint_summary,
        "comparison": comparison,
    }
    write_json(out_dir / "offline_replay_results.json", payload)
    _write_csv(
        out_dir / "offline_replay_summary.csv",
        summary_rows,
        fieldnames=[
            "model",
            "num_sequences",
            "requested_steps_per_sequence",
            "min_valid_horizon_reference",
            "mean_chunk_l1",
            "mean_first_action_l1",
            "mean_tail_action_l1",
            "policy_dir",
            "video_dir",
        ],
    )
    _write_csv(
        out_dir / "offline_replay_step_metrics.csv",
        basic_rows + joint_rows,
        fieldnames=[
            "model",
            "sequence_id",
            "dataset_index",
            "episode_index",
            "frame_index",
            "local_step",
            "chunk_l1",
            "first_action_l1",
            "tail_action_l1",
            "valid_horizon",
            "sample_valid_horizon",
            "was_clipped",
            "below_min_valid_horizon",
        ],
    )
    _write_csv(
        out_dir / "offline_replay_sequence_metrics.csv",
        basic_seq_rows + joint_seq_rows,
        fieldnames=[
            "model",
            "sequence_id",
            "start_index",
            "episode_index",
            "frame_index",
            "steps",
            "requested_steps",
            "mean_chunk_l1",
            "mean_valid_horizon",
            "min_valid_horizon_observed",
            "video_path",
        ],
    )

    if cfg.wandb_enable and cfg.wandb_mode != "disabled":
        import wandb
        wandb.init(project=cfg.wandb_project, name="offline_replay_visualization", mode=cfg.wandb_mode, dir=str(cfg.output_dir / "wandb"))
        wandb.log({
            "replay/basic_mean_chunk_l1": basic_summary["mean_chunk_l1"],
            "replay/joint_mean_chunk_l1": joint_summary["mean_chunk_l1"],
            "replay/joint_minus_basic_mean_chunk_l1": comparison["joint_minus_basic_mean_chunk_l1"],
            "replay/relative_improvement_of_joint": comparison["relative_improvement_of_joint"],
        })
        wandb.finish()

    print(f"[replay] Saved summary to {out_dir / 'offline_replay_summary.csv'}")
    print(f"[replay] Saved sequence metrics to {out_dir / 'offline_replay_sequence_metrics.csv'}")
    print(f"[replay] Saved videos under {out_dir / 'videos'}")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import argparse
    from .constants import DEFAULT_PROJECT

    p = argparse.ArgumentParser(prog="python -m crossenv_act.offline_replay")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--cuda-id", default="0")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--basic-policy", required=True)
    p.add_argument("--joint-policy", required=True)
    p.add_argument("--num-sequences", type=int, default=4, help="Auto-select this many episodes without scanning the full dataset when --sequence-ids is not set.")
    p.add_argument("--sequence-ids", default=None, help="Comma-separated episode ids to visualize, e.g. 2,10,17,21. No full candidate scan is performed.")
    p.add_argument("--start-indices", default=None, help="Comma-separated exact dataset start indices to visualize.")
    p.add_argument("--list-candidates-only", action="store_true", help="Deprecated: now only writes selected replay windows and exits.")
    p.add_argument("--fast-generate", dest="fast_generate", action="store_true", default=True, help="Deprecated compatibility flag; generation is always direct and no full candidate scan is performed.")
    p.add_argument("--no-fast-generate", dest="fast_generate", action="store_false", help="Deprecated compatibility flag; ignored.")
    p.add_argument("--steps-per-sequence", type=int, default=32)
    p.add_argument("--candidate-stride", type=int, default=64, help="Spacing between safe candidate windows inside each episode.")
    p.add_argument("--min-frame-index", type=int, default=5, help="Avoid episode beginnings when constructing candidates.")
    p.add_argument("--end-margin", type=int, default=5, help="Avoid episode endings when constructing candidates.")
    p.add_argument("--min-valid-horizon", type=int, default=32, help="Require at least this many valid non-padding action horizons for every visualized frame.")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--tolerance-s", type=float, default=1e-4)
    p.add_argument("--wandb-enable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    raise SystemExit(replay_entry(p.parse_args()))
