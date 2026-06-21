from __future__ import annotations

import csv
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data_prep import prepare_all, root_for_mode
from .paths import resolve_path
from .train import pretrained_dir, runtime_env, write_json

ACTION = "action"


class EvaluationError(RuntimeError):
    """Raised when zero-shot evaluation is structurally invalid."""


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def _require_arg(args: Any, name: str) -> Any:
    if not hasattr(args, name):
        raise EvaluationError(
            f"CLI argument '{name}' is missing. Your cli.py is stale or the patch was not applied cleanly. "
            f"Re-apply the zero-shot strict patch and run `pip install -e .`."
        )
    return getattr(args, name)


def _policy_chunk_size(policy: Any) -> int:
    cfg = getattr(policy, "config", None)
    value = getattr(cfg, "chunk_size", None)
    if isinstance(value, int):
        return value
    try:
        indices = cfg.action_delta_indices
        return len(indices) if indices is not None else 1
    except Exception:
        return 1


def _larger_chunk_config(policy_a: Any, policy_b: Any) -> Any:
    return policy_a.config if _policy_chunk_size(policy_a) >= _policy_chunk_size(policy_b) else policy_b.config


def make_lerobot_dataset(repo_id: str, root: Path, revision: str, policy_config: Any, tolerance_s: float):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(root), revision=revision)
    delta_timestamps = resolve_delta_timestamps(policy_config, ds_meta)
    kwargs: dict[str, Any] = {
        "root": str(root),
        "episodes": None,
        "delta_timestamps": delta_timestamps,
        "revision": revision,
        "tolerance_s": tolerance_s,
    }
    params = inspect.signature(LeRobotDataset.__init__).parameters
    # Different LeRobot releases expose slightly different dataset kwargs. We only
    # pass kwargs that the installed version supports, and we do not silently catch
    # constructor errors.
    if "return_uint8" in params:
        kwargs["return_uint8"] = True
    if "download_videos" in params:
        kwargs["download_videos"] = True
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    dataset = LeRobotDataset(repo_id, **kwargs)
    return dataset, ds_meta, delta_timestamps


def load_policy(policy_path: str | Path, device: torch.device):
    from lerobot.policies.factory import get_policy_class

    pdir = pretrained_dir(policy_path)
    if not pdir.exists():
        raise EvaluationError(f"Policy directory does not exist: {pdir}")
    policy_cls = get_policy_class("act")
    policy = policy_cls.from_pretrained(pdir)
    policy.to(device)
    policy.eval()
    return policy, pdir


def make_preprocessor(policy: Any, policy_dir: Path, dataset_stats: dict[str, Any]):
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_dir),
        dataset_stats=dataset_stats,
    )
    return preprocessor


def predict_chunk(policy: Any, batch: dict[str, Any]) -> torch.Tensor:
    policy.eval()
    if hasattr(policy, "predict_action_chunk"):
        pred = policy.predict_action_chunk(batch)
    else:
        # Compatibility fallback for older ACTPolicy implementations.
        if getattr(policy.config, "image_features", None):
            batch = dict(batch)
            batch["observation.images"] = [batch[k] for k in policy.config.image_features]
        pred = policy.model(batch)[0]

    if not torch.is_tensor(pred):
        raise EvaluationError(f"Policy prediction is not a tensor: {type(pred)}")
    if pred.ndim != 3:
        raise EvaluationError(f"Expected action chunk prediction shape (B,T,A), got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all().item():
        raise EvaluationError("Policy prediction contains NaN or Inf. Zero-shot inference is invalid.")
    return pred


def _validate_action_tensors(pred: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ACTION not in batch:
        keys = sorted(batch.keys())
        raise EvaluationError(
            "Evaluation batch does not contain standardized 'action'. "
            f"Prepared splitD schema is invalid. Available keys: {keys}"
        )
    target = batch[ACTION]
    if not torch.is_tensor(target):
        raise EvaluationError(f"Batch 'action' is not a tensor: {type(target)}")
    if target.ndim != 3:
        raise EvaluationError(f"Expected target action shape (B,T,A), got {tuple(target.shape)}")
    if pred.shape[0] != target.shape[0]:
        raise EvaluationError(f"Batch size mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.shape[-1] != target.shape[-1]:
        raise EvaluationError(f"Action dim mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    horizon = min(pred.shape[1], target.shape[1])
    if horizon <= 0:
        raise EvaluationError(f"No overlapping action horizon: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    pred = pred[:, :horizon]
    target = target[:, :horizon]

    if "action_is_pad" in batch and torch.is_tensor(batch["action_is_pad"]):
        mask = ~batch["action_is_pad"][:, :horizon].bool()
    else:
        mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    if mask.shape != target.shape[:2]:
        raise EvaluationError(f"action_is_pad shape mismatch: mask {tuple(mask.shape)}, action {tuple(target.shape)}")
    return pred, target, mask


def action_error_stats_from_batch(pred: torch.Tensor, batch: dict[str, Any], success_threshold: float) -> dict[str, Any]:
    pred, target, valid_step_mask = _validate_action_tensors(pred, batch)
    abs_err = (pred - target).abs()
    batch_size, horizon, action_dim = abs_err.shape
    valid = valid_step_mask.unsqueeze(-1).to(dtype=abs_err.dtype)

    overall_abs_error_sum = float((abs_err * valid).sum().detach().cpu().item())
    valid_steps = valid_step_mask.sum(dim=1)  # (B,)
    valid_action_count = int(valid_step_mask.sum().detach().cpu().item()) * action_dim
    valid_samples_mask = valid_steps > 0
    valid_samples = int(valid_samples_mask.sum().detach().cpu().item())

    if valid_samples == 0 or valid_action_count == 0:
        raise EvaluationError("All evaluated action chunks are padding; cannot compute zero-shot action error.")

    per_sample_sum = (abs_err * valid).sum(dim=(1, 2))
    per_sample_den = (valid_steps.to(abs_err.dtype) * action_dim).clamp_min(1)
    per_sample_l1 = per_sample_sum / per_sample_den
    success_count = int(((per_sample_l1 <= success_threshold) & valid_samples_mask).sum().detach().cpu().item())

    horizon_rows: list[dict[str, Any]] = []
    for h in range(horizon):
        h_mask = valid_step_mask[:, h]
        h_valid_samples = int(h_mask.sum().detach().cpu().item())
        h_action_count = h_valid_samples * action_dim
        h_abs_sum = float((abs_err[:, h, :] * h_mask.unsqueeze(-1).to(abs_err.dtype)).sum().detach().cpu().item())
        horizon_rows.append(
            {
                "horizon": h,
                "abs_error_sum": h_abs_sum,
                "valid_action_count": h_action_count,
                "valid_samples": h_valid_samples,
            }
        )

    return {
        "overall_abs_error_sum": overall_abs_error_sum,
        "action_value_count": valid_action_count,
        "success_count": success_count,
        "valid_samples": valid_samples,
        "batch_size": batch_size,
        "horizon": horizon,
        "action_dim": action_dim,
        "horizon_rows": horizon_rows,
    }


def _tail_horizon_l1(horizon_rows: list[dict[str, Any]], tail_fraction: float = 0.25) -> float | None:
    rows = [r for r in horizon_rows if int(r.get("valid_action_count", 0)) > 0]
    if not rows:
        return None
    start = max(0, int(math.floor(len(rows) * (1.0 - tail_fraction))))
    tail = rows[start:]
    abs_sum = sum(float(r["abs_error_sum"]) for r in tail)
    count = sum(int(r["valid_action_count"]) for r in tail)
    return abs_sum / count if count > 0 else None


def evaluate_loaded_policy(
    model_name: str,
    policy: Any,
    policy_dir: Path,
    dataset: Any,
    dataset_stats: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    success_threshold: float,
) -> dict[str, Any]:
    preprocessor = make_preprocessor(policy, policy_dir, dataset_stats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    total_loader_batches = len(loader)
    if total_loader_batches == 0:
        raise EvaluationError("Evaluation DataLoader has zero batches. Check prepared splitD dataset.")
    limit_batches = total_loader_batches if max_batches is None else min(max_batches, total_loader_batches)
    if limit_batches <= 0:
        raise EvaluationError(f"max_batches={max_batches} leaves no batches to evaluate.")

    total_abs_error = 0.0
    total_action_values = 0
    total_success = 0
    total_samples = 0
    batches = 0
    horizon_accum: dict[int, dict[str, Any]] = {}

    progress = tqdm(total=limit_batches, desc=f"Evaluating {model_name}", unit="batch")
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= limit_batches:
                    break
                meta = getattr(dataset, "meta", None)
                camera_keys = getattr(meta, "camera_keys", [])
                for cam_key in camera_keys:
                    if cam_key in batch and torch.is_tensor(batch[cam_key]) and batch[cam_key].dtype == torch.uint8:
                        batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

                batch = preprocessor(batch)
                batch = move_to_device(batch, device)
                pred = predict_chunk(policy, batch)
                stats = action_error_stats_from_batch(pred, batch, success_threshold=success_threshold)

                total_abs_error += float(stats["overall_abs_error_sum"])
                total_action_values += int(stats["action_value_count"])
                total_success += int(stats["success_count"])
                total_samples += int(stats["valid_samples"])
                batches += 1

                for row in stats["horizon_rows"]:
                    h = int(row["horizon"])
                    acc = horizon_accum.setdefault(
                        h,
                        {"horizon": h, "abs_error_sum": 0.0, "valid_action_count": 0, "valid_samples": 0},
                    )
                    acc["abs_error_sum"] += float(row["abs_error_sum"])
                    acc["valid_action_count"] += int(row["valid_action_count"])
                    acc["valid_samples"] += int(row["valid_samples"])
                progress.update(1)
    finally:
        progress.close()

    if batches != limit_batches:
        raise EvaluationError(f"Evaluation ended early for {model_name}: {batches}/{limit_batches} batches processed.")
    if total_action_values <= 0 or total_samples <= 0:
        raise EvaluationError(f"No valid action values were evaluated for {model_name}.")

    horizon_rows: list[dict[str, Any]] = []
    for h in sorted(horizon_accum):
        acc = horizon_accum[h]
        denom = int(acc["valid_action_count"])
        if denom <= 0:
            continue
        horizon_rows.append(
            {
                "horizon": h,
                "action_l1_loss": float(acc["abs_error_sum"]) / denom,
                "abs_error_sum": float(acc["abs_error_sum"]),
                "valid_action_count": denom,
                "valid_samples": int(acc["valid_samples"]),
            }
        )
    if not horizon_rows:
        raise EvaluationError(f"No valid chunk horizon rows were produced for {model_name}.")

    action_l1 = total_abs_error / total_action_values
    proxy_success_rate = total_success / total_samples
    return {
        "policy_dir": str(policy_dir),
        "batches": batches,
        "total_loader_batches": total_loader_batches,
        "is_partial_eval": max_batches is not None and limit_batches < total_loader_batches,
        "max_batches": max_batches,
        "samples": total_samples,
        "action_value_count": total_action_values,
        "action_l1_loss": action_l1,
        "proxy_success_rate": proxy_success_rate,
        "proxy_success_count": total_success,
        "proxy_success_threshold": success_threshold,
        "chunk_tail_action_l1_loss": _tail_horizon_l1(horizon_rows),
        "horizon_errors": horizon_rows,
    }


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.2f}"
    return value


def write_overall_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    priority = [
        "model",
        "action_l1_loss",
        "chunk_tail_action_l1_loss",
        "proxy_success_rate",
        "proxy_success_count",
        "samples",
        "action_value_count",
        "proxy_success_threshold",
        "batches",
        "total_loader_batches",
        "is_partial_eval",
        "max_batches",
        "policy_dir",
    ]
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    fieldnames = [k for k in priority if k in keys] + sorted(k for k in keys if k not in priority and k != "horizon_errors")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_value(row.get(k)) for k in fieldnames})


def write_horizon_csv(path: Path, model_to_rows: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = ["model", "horizon", "action_l1_loss", "valid_action_count", "valid_samples"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, rows in model_to_rows.items():
            for row in rows:
                writer.writerow(
                    {
                        "model": model,
                        "horizon": row["horizon"],
                        "action_l1_loss": _format_csv_value(row["action_l1_loss"]),
                        "valid_action_count": row["valid_action_count"],
                        "valid_samples": row["valid_samples"],
                    }
                )


def write_horizon_plot(path: Path, model_to_rows: dict[str, list[dict[str, Any]]]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # optional dependency
        tqdm.write(f"[eval warning] Could not import matplotlib; skip horizon plot: {exc}")
        return False

    plt.figure(figsize=(8, 5))
    for model, rows in model_to_rows.items():
        xs = [int(r["horizon"]) for r in rows]
        ys = [float(r["action_l1_loss"]) for r in rows]
        plt.plot(xs, ys, marker="o", markersize=2, linewidth=1.5, label=model)
    plt.xlabel("Action chunk horizon")
    plt.ylabel("Normalized Action L1 Loss")
    plt.title("Zero-shot splitD action chunk horizon error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def _public_result(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "horizon_errors"}


def eval_entry(args) -> int:
    success_threshold = float(_require_arg(args, "success_threshold"))
    if success_threshold <= 0:
        raise EvaluationError(f"--success-threshold must be positive, got {success_threshold}")

    env = runtime_env(args.data_dir, args.output_dir, args.cuda_id)
    os.environ.update(env)
    output_root = resolve_path(args.output_dir)
    prepare_all(
        args.data_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force_download=args.force_download,
        force_prepare=args.force_prepare,
    )
    dataset_root = root_for_mode(args.data_dir, "D")
    device = torch.device("cpu" if str(args.cuda_id).lower() in {"cpu", "none", "-1"} or not torch.cuda.is_available() else "cuda")

    basic_policy, basic_dir = load_policy(args.basic_policy, device)
    joint_policy, joint_dir = load_policy(args.joint_policy, device)
    dataset_config = _larger_chunk_config(basic_policy, joint_policy)
    dataset, ds_meta, delta_timestamps = make_lerobot_dataset(
        args.repo_id,
        dataset_root,
        args.revision,
        dataset_config,
        args.tolerance_s,
    )
    full_dataset_len = len(dataset)
    if full_dataset_len == 0:
        raise EvaluationError(f"Prepared splitD dataset is empty: {dataset_root}")

    if args.max_samples is not None and 0 < args.max_samples < len(dataset):
        subset = Subset(dataset, list(range(args.max_samples)))
        subset.meta = dataset.meta  # type: ignore[attr-defined]
        dataset = subset

    if args.max_batches is not None:
        tqdm.write(
            f"[eval notice] --max-batches={args.max_batches} is a smoke/partial evaluation. "
            "Remove it for the full zero-shot result used in the report."
        )

    results: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "test_env": "D",
        "dataset_root": str(dataset_root),
        "full_dataset_len": full_dataset_len,
        "effective_dataset_len": len(dataset),
        "metric": "offline normalized action L1, proxy success, and chunk-horizon error on splitD",
        "proxy_success_threshold": success_threshold,
        "delta_timestamps": delta_timestamps,
        "basic_chunk_size": _policy_chunk_size(basic_policy),
        "joint_chunk_size": _policy_chunk_size(joint_policy),
    }
    results["basic"] = evaluate_loaded_policy(
        "act_env_b",
        basic_policy,
        basic_dir,
        dataset,
        ds_meta.stats,
        device,
        args.batch_size,
        args.num_workers,
        args.max_batches,
        success_threshold,
    )
    results["joint"] = evaluate_loaded_policy(
        "act_env_abc",
        joint_policy,
        joint_dir,
        dataset,
        ds_meta.stats,
        device,
        args.batch_size,
        args.num_workers,
        args.max_batches,
        success_threshold,
    )

    b_l1 = float(results["basic"]["action_l1_loss"])
    j_l1 = float(results["joint"]["action_l1_loss"])
    b_sr = float(results["basic"]["proxy_success_rate"])
    j_sr = float(results["joint"]["proxy_success_rate"])
    b_tail = results["basic"].get("chunk_tail_action_l1_loss")
    j_tail = results["joint"].get("chunk_tail_action_l1_loss")
    results["comparison"] = {
        "joint_minus_basic_action_l1": j_l1 - b_l1,
        "relative_improvement_of_joint_l1": (b_l1 - j_l1) / b_l1 if b_l1 != 0 else None,
        "joint_minus_basic_proxy_success_rate": j_sr - b_sr,
        "joint_minus_basic_chunk_tail_action_l1": (j_tail - b_tail) if b_tail is not None and j_tail is not None else None,
    }

    out_dir = output_root / "zero_shot_env_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "offline_eval_results.json"
    csv_path = out_dir / "offline_eval_results.csv"
    horizon_csv_path = out_dir / "chunk_horizon_error.csv"
    horizon_png_path = out_dir / "chunk_horizon_error.png"

    write_json(json_path, results)
    overall_rows = [
        {"model": "basic", **_public_result(results["basic"])},
        {"model": "joint", **_public_result(results["joint"])},
    ]
    write_overall_csv(csv_path, overall_rows)
    write_horizon_csv(
        horizon_csv_path,
        {
            "basic": results["basic"]["horizon_errors"],
            "joint": results["joint"]["horizon_errors"],
        },
    )
    plot_ok = write_horizon_plot(
        horizon_png_path,
        {
            "basic": results["basic"]["horizon_errors"],
            "joint": results["joint"]["horizon_errors"],
        },
    )

    if args.wandb_enable:
        import wandb

        wandb.init(project=args.wandb_project, name="zero_shot_env_d", mode=args.wandb_mode, dir=str(output_root / "wandb"))
        wandb.log(
            {
                "eval/basic_action_l1_loss": results["basic"]["action_l1_loss"],
                "eval/joint_action_l1_loss": results["joint"]["action_l1_loss"],
                "eval/basic_proxy_success_rate": results["basic"]["proxy_success_rate"],
                "eval/joint_proxy_success_rate": results["joint"]["proxy_success_rate"],
                "eval/basic_chunk_tail_action_l1_loss": results["basic"]["chunk_tail_action_l1_loss"],
                "eval/joint_chunk_tail_action_l1_loss": results["joint"]["chunk_tail_action_l1_loss"],
                "eval/joint_minus_basic_action_l1": results["comparison"]["joint_minus_basic_action_l1"],
                "eval/joint_minus_basic_proxy_success_rate": results["comparison"]["joint_minus_basic_proxy_success_rate"],
                "eval/is_partial_eval": bool(results["basic"]["is_partial_eval"] or results["joint"]["is_partial_eval"]),
            }
        )
        if plot_ok:
            wandb.log({"eval/chunk_horizon_error": wandb.Image(str(horizon_png_path))})
        wandb.finish()

    tqdm.write(f"[eval] Saved results to {json_path}")
    tqdm.write(f"[eval] Saved overall CSV to {csv_path}")
    tqdm.write(f"[eval] Saved chunk horizon CSV to {horizon_csv_path}")
    if plot_ok:
        tqdm.write(f"[eval] Saved chunk horizon plot to {horizon_png_path}")
    if results["basic"]["is_partial_eval"] or results["joint"]["is_partial_eval"]:
        tqdm.write("[eval notice] This was a PARTIAL evaluation because --max-batches was set. Do not report it as full splitD zero-shot.")
    tqdm.write(json.dumps(results["comparison"], indent=2, ensure_ascii=False))
    return 0
