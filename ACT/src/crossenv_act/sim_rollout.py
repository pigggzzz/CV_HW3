from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .constants import DEFAULT_PROJECT
from .data_prep import prepare_all, root_for_mode
from .paths import resolve_path
from .train import pretrained_dir, runtime_env, write_json


class SimRolloutError(RuntimeError):
    pass


@dataclass
class CalvinPaths:
    calvin_root: Path | None
    dataset_path: Path
    conf_dir: Path


def _round2(x: Any) -> Any:
    if isinstance(x, float):
        return f"{x:.2f}"
    if isinstance(x, (np.floating,)):
        return f"{float(x):.2f}"
    return x


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "model", "num_sequences", "max_subtasks", "max_steps", "successes", "attempted_subtasks",
        "subtask_success_rate", "full_sequence_success_rate", "mean_successes_per_sequence",
        "success_at_1", "success_at_2", "success_at_3", "success_at_4", "success_at_5",
        "video_dir", "policy_dir", "calvin_dataset_path",
    ]
    keys = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _round2(row.get(k, "")) for k in keys})


def _import_or_raise(module: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:
        raise SimRolloutError(
            f"Cannot import {module!r}. This command requires the official CALVIN simulator, not only the "
            "LeRobot-format dataset. Install CALVIN first, e.g. clone https://github.com/mees/calvin "
            "with submodules, run its install.sh, and set CALVIN_ROOT. Original error: " + repr(exc)
        ) from exc


def resolve_calvin_paths(args) -> CalvinPaths:
    calvin_root = None
    if getattr(args, "calvin_root", None):
        calvin_root = resolve_path(args.calvin_root)
    elif os.environ.get("CALVIN_ROOT"):
        calvin_root = resolve_path(os.environ["CALVIN_ROOT"])

    if getattr(args, "calvin_dataset_path", None):
        dataset_path = resolve_path(args.calvin_dataset_path)
    elif calvin_root is not None:
        candidates = [
            calvin_root / "dataset" / "task_D_D",
            calvin_root / "dataset" / "task_ABC_D",
            calvin_root / "dataset" / "debug",
            calvin_root / "dataset",
        ]
        dataset_path = next((p for p in candidates if (p / "validation").exists()), candidates[0])
    else:
        raise SimRolloutError(
            "Please provide --calvin-dataset-path or set CALVIN_ROOT. For official CALVIN evaluation this path "
            "must contain a validation/ folder, e.g. $CALVIN_ROOT/dataset/task_D_D."
        )

    if not (dataset_path / "validation").exists():
        raise SimRolloutError(
            f"CALVIN dataset_path={dataset_path} does not contain validation/. This is not the same as the "
            "LeRobot xiaoma26/calvin-lerobot directory. Download official CALVIN simulator data, e.g. "
            "cd $CALVIN_ROOT/dataset && sh download_data.sh D or ABCD."
        )

    if getattr(args, "calvin_conf_dir", None):
        conf_dir = resolve_path(args.calvin_conf_dir)
    elif calvin_root is not None:
        conf_dir = calvin_root / "calvin_models" / "conf"
    else:
        # Try to infer from installed calvin_agent package.
        try:
            calvin_agent = _import_or_raise("calvin_agent")
            conf_dir = Path(calvin_agent.__file__).resolve().parents[1] / "conf"
        except Exception as exc:
            raise SimRolloutError("Could not infer CALVIN conf dir. Pass --calvin-conf-dir explicitly.") from exc

    if not (conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml").exists():
        raise SimRolloutError(
            f"CALVIN conf_dir={conf_dir} does not look valid; missing callbacks/rollout/tasks/new_playtable_tasks.yaml."
        )
    if not (conf_dir / "annotations" / "new_playtable_validation.yaml").exists():
        raise SimRolloutError(
            f"CALVIN conf_dir={conf_dir} does not look valid; missing annotations/new_playtable_validation.yaml."
        )
    return CalvinPaths(calvin_root=calvin_root, dataset_path=dataset_path, conf_dir=conf_dir)


def check_calvin_installation(args) -> dict[str, Any]:
    paths = resolve_calvin_paths(args)
    imports = {}
    for module in [
        "calvin_env.envs.play_table_env",
        "calvin_agent.evaluation.multistep_sequences",
        "calvin_agent.evaluation.utils",
        "hydra",
        "omegaconf",
        "pybullet",
    ]:
        try:
            importlib.import_module(module)
            imports[module] = "ok"
        except Exception as exc:
            imports[module] = f"missing: {exc!r}"
    payload = {
        "calvin_root": str(paths.calvin_root) if paths.calvin_root else None,
        "calvin_dataset_path": str(paths.dataset_path),
        "calvin_conf_dir": str(paths.conf_dir),
        "validation_exists": (paths.dataset_path / "validation").exists(),
        "imports": imports,
    }
    missing = {k: v for k, v in imports.items() if v != "ok"}
    if missing:
        raise SimRolloutError("CALVIN simulator dependency check failed:\n" + json.dumps(payload, indent=2))
    return payload


def load_policy_and_processors(policy_path: str | Path, device: torch.device, data_dir: str | Path, repo_id: str, revision: str):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    pdir = pretrained_dir(policy_path)
    policy_cls = get_policy_class("act")
    policy = policy_cls.from_pretrained(pdir)
    policy.to(device)
    policy.eval()

    # Use prepared splitD statistics for input normalization and action unnormalization.
    d_root = root_for_mode(data_dir, "D")
    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(d_root), revision=revision)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pdir),
        dataset_stats=ds_meta.stats,
    )
    return policy, preprocessor, postprocessor, pdir, ds_meta


def _as_chw_float_image(x: Any, expected_chw: tuple[int, int, int] | None = None) -> torch.Tensor:
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise SimRolloutError(f"Expected image with 3 dims HWC/CHW, got shape {arr.shape}")
    # Most CALVIN observations are HWC RGB uint8.
    if arr.shape[-1] in (1, 3, 4):
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
    elif arr.shape[0] in (1, 3, 4):
        if arr.shape[0] == 4:
            arr = arr[:3]
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    else:
        raise SimRolloutError(f"Could not infer channel axis for image shape {arr.shape}")
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    if expected_chw is not None:
        c, h, w = expected_chw
        if tensor.shape[0] != c:
            if tensor.shape[0] == 1 and c == 3:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] >= c:
                tensor = tensor[:c]
            else:
                pad = torch.zeros(c - tensor.shape[0], tensor.shape[1], tensor.shape[2], dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
        if tuple(tensor.shape[-2:]) != (h, w):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
    return tensor.unsqueeze(0)


def _pick_calvin_rgb(obs: dict[str, Any], feature_key: str):
    rgb_obs = obs.get("rgb_obs", {}) if isinstance(obs, dict) else {}
    low = feature_key.lower()
    if "gripper" in low or "wrist" in low:
        candidates = ["rgb_gripper", "gripper", "wrist", "rgb_wrist"]
    else:
        candidates = ["rgb_static", "static", "image", "rgb", "camera", "front"]
    for c in candidates:
        if isinstance(rgb_obs, dict) and c in rgb_obs:
            return rgb_obs[c]
        if c in obs:
            return obs[c]
    if isinstance(rgb_obs, dict) and rgb_obs:
        # Prefer static if available, otherwise first RGB-like image.
        for k in sorted(rgb_obs.keys()):
            if "static" in k:
                return rgb_obs[k]
        return next(iter(rgb_obs.values()))
    raise SimRolloutError(
        f"Could not find an RGB observation for policy image feature {feature_key!r}. "
        f"Available obs keys: {list(obs.keys())}; rgb_obs keys: {list(rgb_obs.keys()) if isinstance(rgb_obs, dict) else None}"
    )


def _vector_feature(obs: dict[str, Any], feature_key: str, dim: int) -> torch.Tensor:
    candidates = []
    if feature_key == "observation.state" or feature_key.endswith(".state"):
        candidates = ["robot_obs", "state", "proprio", "proprioception"]
    elif "environment_state" in feature_key:
        candidates = ["scene_obs", "environment_state"]
    else:
        candidates = [feature_key, feature_key.split(".")[-1], "robot_obs", "state"]
    value = None
    for c in candidates:
        if c in obs:
            value = obs[c]
            break
    if value is None:
        raise SimRolloutError(f"Could not find vector feature {feature_key!r}. Available obs keys: {list(obs.keys())}")
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < dim:
        arr = np.pad(arr, (0, dim - arr.shape[0]))
    elif arr.shape[0] > dim:
        arr = arr[:dim]
    return torch.from_numpy(arr).float().unsqueeze(0)


def calvin_obs_to_lerobot_batch(obs: dict[str, Any], policy, device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    input_features = getattr(policy.config, "input_features", {}) or {}
    for key, ft in input_features.items():
        ftype = getattr(ft, "type", None)
        ftype_name = getattr(ftype, "name", str(ftype)).upper()
        shape = tuple(getattr(ft, "shape", ()))
        if "VISUAL" in ftype_name or key.startswith("observation.images"):
            # LeRobot policy features use C,H,W for visual input.
            expected = shape if len(shape) == 3 else None
            batch[key] = _as_chw_float_image(_pick_calvin_rgb(obs, key), expected).to(device)
        elif key == "observation.state" or "STATE" in ftype_name or "ENV" in ftype_name:
            dim = int(shape[0]) if shape else 1
            batch[key] = _vector_feature(obs, key, dim).to(device)
    if not batch:
        raise SimRolloutError("Could not construct any policy input feature from CALVIN observation.")
    return batch


def postprocess_action(postprocessor, action: torch.Tensor, *, allow_unprocessed: bool = False) -> np.ndarray:
    # LeRobot postprocessors usually expect and return a transition/batch dict containing 'action'.
    errors = []
    for candidate in ({"action": action}, action):
        try:
            out = postprocessor(candidate)
            if isinstance(out, dict) and "action" in out:
                act = out["action"]
            else:
                act = out
            if torch.is_tensor(act):
                act = act.detach().cpu().numpy()
            arr = np.asarray(act, dtype=np.float32)
            if arr.ndim >= 2:
                arr = arr[0]
            return arr.reshape(-1)
        except Exception as exc:
            errors.append(repr(exc))
    if allow_unprocessed:
        arr = action.detach().cpu().numpy()
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1).astype(np.float32)
    raise SimRolloutError(
        "Could not postprocess ACT action back to CALVIN action space. This usually means the LeRobot "
        "processor API differs from the package version. Re-run with --allow-unprocessed-actions only for debugging, "
        "not for final success-rate reporting. Errors: " + " | ".join(errors)
    )


class LeRobotACTCalvinWrapper:
    def __init__(self, name: str, policy_path: str | Path, device: torch.device, args):
        self.name = name
        self.policy, self.preprocessor, self.postprocessor, self.policy_dir, self.ds_meta = load_policy_and_processors(
            policy_path, device, args.data_dir, args.repo_id, args.revision
        )
        self.device = device
        self.allow_unprocessed_actions = bool(getattr(args, "allow_unprocessed_actions", False))

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @torch.no_grad()
    def step(self, obs: dict[str, Any], goal: str | None = None) -> np.ndarray:
        batch = calvin_obs_to_lerobot_batch(obs, self.policy, self.device)
        batch = self.preprocessor(batch)
        if hasattr(self.policy, "select_action"):
            action = self.policy.select_action(batch)
        else:
            action = self.policy.predict_action_chunk(batch)[:, 0]
        return postprocess_action(self.postprocessor, action, allow_unprocessed=self.allow_unprocessed_actions)


def _find_conf_and_oracle(paths: CalvinPaths):
    hydra = _import_or_raise("hydra")
    OmegaConf = _import_or_raise("omegaconf").OmegaConf
    task_cfg = OmegaConf.load(paths.conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(paths.conf_dir / "annotations" / "new_playtable_validation.yaml")
    return task_oracle, val_annotations


def _make_env(dataset_path: Path, show_gui: bool):
    play_table_env = _import_or_raise("calvin_env.envs.play_table_env")
    val_folder = dataset_path / "validation"
    return play_table_env.get_env(val_folder, show_gui=show_gui)


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> str | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return str(path)
    except Exception:
        # Fallback to gif; this avoids losing qualitative visualization when ffmpeg is unavailable.
        try:
            import imageio.v2 as imageio
            gif = path.with_suffix(".gif")
            imageio.mimsave(gif, frames, fps=max(1, min(fps, 15)))
            return str(gif)
        except Exception as exc:
            raise SimRolloutError(f"Failed to save rollout video {path}: {exc!r}") from exc


def _render_frame(env) -> np.ndarray | None:
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        try:
            frame = env.render()
        except Exception:
            return None
    except Exception:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[..., :3].astype(np.uint8)
    return None


def _rollout_subtask(env, model: LeRobotACTCalvinWrapper, task_oracle, val_annotations, subtask: str, max_steps: int, *, record: bool, video_every_n: int, debug: bool):
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    obs = env.get_obs()
    frames: list[np.ndarray] = []
    for step in range(max_steps):
        action = model.step(obs, lang_annotation)
        if not np.all(np.isfinite(action)):
            raise SimRolloutError(f"Policy {model.name} produced non-finite action at step {step}: {action}")
        obs, _, _, current_info = env.step(action)
        if record and step % max(1, video_every_n) == 0:
            frame = _render_frame(env)
            if frame is not None:
                frames.append(frame)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, step + 1, frames
    return False, max_steps, frames


def _count_success_levels(success_counts: list[int], max_subtasks: int) -> dict[str, float]:
    n = max(1, len(success_counts))
    out = {}
    for k in range(1, max_subtasks + 1):
        out[f"success_at_{k}"] = sum(c >= k for c in success_counts) / n
    return out


def evaluate_model_rollout(model_name: str, policy_path: str | Path, args, paths: CalvinPaths, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequences_mod = _import_or_raise("calvin_agent.evaluation.multistep_sequences")
    utils_mod = _import_or_raise("calvin_agent.evaluation.utils")
    task_oracle, val_annotations = _find_conf_and_oracle(paths)
    env = _make_env(paths.dataset_path, show_gui=bool(args.show_gui))
    model = LeRobotACTCalvinWrapper(model_name, policy_path, device, args)

    sequences = list(sequences_mod.get_sequences(int(args.num_sequences)))
    rows: list[dict[str, Any]] = []
    success_counts: list[int] = []
    total_attempted = 0
    total_success = 0
    video_dir = resolve_path(args.output_dir) / "sim_rollout_env_d" / "videos" / model_name

    pbar = tqdm(enumerate(sequences), total=len(sequences), desc=f"Simulator rollout {model_name}", unit="seq")
    for seq_idx, (initial_state, eval_sequence) in pbar:
        eval_sequence = list(eval_sequence)[: int(args.max_subtasks)]
        robot_obs, scene_obs = utils_mod.get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        solved = 0
        steps_total = 0
        sequence_frames: list[np.ndarray] = []
        per_subtask = []
        for subtask in eval_sequence:
            record = seq_idx < int(args.record_videos)
            ok, steps, frames = _rollout_subtask(
                env,
                model,
                task_oracle,
                val_annotations,
                subtask,
                int(args.max_steps),
                record=record,
                video_every_n=int(args.video_every_n),
                debug=bool(args.debug),
            )
            steps_total += steps
            total_attempted += 1
            if record:
                sequence_frames.extend(frames)
            per_subtask.append({"subtask": subtask, "success": bool(ok), "steps": int(steps)})
            if ok:
                solved += 1
                total_success += 1
            else:
                # Official CALVIN long-horizon metric stops after first failed subtask.
                break
        video_path = None
        if seq_idx < int(args.record_videos):
            video_path = _save_video(sequence_frames, video_dir / f"sequence_{seq_idx:04d}.mp4", int(args.video_fps))
        success_counts.append(solved)
        row = {
            "model": model_name,
            "sequence_index": seq_idx,
            "success_count": solved,
            "max_subtasks": int(args.max_subtasks),
            "full_sequence_success": solved >= int(args.max_subtasks),
            "steps": steps_total,
            "eval_sequence": " -> ".join(eval_sequence),
            "video_path": video_path or "",
            "per_subtask": json.dumps(per_subtask, ensure_ascii=False),
        }
        rows.append(row)
        pbar.set_postfix({"mean_solved": f"{np.mean(success_counts):.2f}", "full_sr": f"{np.mean([c >= int(args.max_subtasks) for c in success_counts]):.2f}"})

    nseq = max(1, len(success_counts))
    summary = {
        "model": model_name,
        "num_sequences": len(success_counts),
        "max_subtasks": int(args.max_subtasks),
        "max_steps": int(args.max_steps),
        "successes": int(total_success),
        "attempted_subtasks": int(total_attempted),
        "subtask_success_rate": float(total_success / max(1, total_attempted)),
        "full_sequence_success_rate": float(sum(c >= int(args.max_subtasks) for c in success_counts) / nseq),
        "mean_successes_per_sequence": float(np.mean(success_counts) if success_counts else 0.0),
        **_count_success_levels(success_counts, int(args.max_subtasks)),
        "video_dir": str(video_dir),
        "policy_dir": str(model.policy_dir),
        "calvin_dataset_path": str(paths.dataset_path),
    }
    return summary, rows


def sim_rollout_entry(args) -> int:
    env = runtime_env(args.data_dir, args.output_dir, args.cuda_id)
    os.environ.update(env)
    if str(args.cuda_id).lower() in {"cpu", "none", "-1"}:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepared splitD is still used for LeRobot processor statistics even though the environment is CALVIN simulator.
    prepare_all(
        args.data_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force_download=args.force_download,
        force_prepare=args.force_prepare,
    )

    paths = resolve_calvin_paths(args)
    install_report = check_calvin_installation(args)
    out_dir = resolve_path(args.output_dir) / "sim_rollout_env_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "calvin_install_check.json", install_report)

    policies = {
        "act_env_b": args.basic_policy,
        "act_env_abc": args.joint_policy,
    }
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for model_name, policy_path in policies.items():
        summary, rows = evaluate_model_rollout(model_name, policy_path, args, paths, device)
        summaries.append(summary)
        episode_rows.extend(rows)

    _write_csv(out_dir / "sim_rollout_results.csv", summaries)
    _write_csv(out_dir / "sim_rollout_episodes.csv", episode_rows)
    write_json(out_dir / "sim_rollout_results.json", {"summary": summaries, "episodes": episode_rows})

    if getattr(args, "wandb_mode", "disabled") != "disabled" and getattr(args, "wandb_enable", True):
        try:
            import wandb
            wandb.init(project=args.wandb_project, name="sim_rollout_env_d", mode=args.wandb_mode, config=vars(args))
            for s in summaries:
                prefix = s["model"]
                wandb.log({f"{prefix}/{k}": v for k, v in s.items() if isinstance(v, (int, float))})
            wandb.finish()
        except Exception as exc:
            print(f"[sim-rollout warning] wandb logging failed: {exc!r}", file=sys.stderr)

    print(f"[sim-rollout] Saved summary: {out_dir / 'sim_rollout_results.csv'}")
    print(f"[sim-rollout] Saved episodes: {out_dir / 'sim_rollout_episodes.csv'}")
    print(f"[sim-rollout] Saved videos under: {out_dir / 'videos'}")
    return 0
