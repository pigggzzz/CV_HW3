from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .constants import DEFAULT_REPO_ID, DEFAULT_REVISION
from .data_prep import prepare_all, root_for_mode
from .paths import resolve_path
from .train_metrics import build_training_metrics_row, write_training_metric_outputs


def runtime_env(data_dir: str | Path, output_dir: str | Path, cuda_id: str | int) -> dict[str, str]:
    env = os.environ.copy()
    data = resolve_path(data_dir)
    out = resolve_path(output_dir)
    env.setdefault("HF_HOME", str(data / "hf_home"))
    env.setdefault("HF_LEROBOT_HOME", str(data / "lerobot_home"))
    if str(cuda_id).lower() not in {"cpu", "none", "-1"}:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("WANDB_DIR", str(out / "wandb"))
    out.mkdir(parents=True, exist_ok=True)
    return env


def require_lerobot_train(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    path = shutil.which("lerobot-train")
    if not path:
        raise FileNotFoundError("Could not find `lerobot-train` in PATH. Activate the ACT/LeRobot environment first.")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command_with_log(cmd: list[str], *, env: dict[str, str], log_path: Path) -> int:
    """Run a command while teeing stdout/stderr to a local log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return int(proc.wait())


def pretrained_dir(run_dir: str | Path) -> Path:
    p = resolve_path(run_dir)
    candidates = [
        p / "checkpoints" / "last" / "pretrained_model",
        p / "pretrained_model",
        p,
    ]
    for c in candidates:
        if c.is_dir() and ((c / "config.json").is_file() or (c / "train_config.json").is_file()):
            return c
    matches = list(p.glob("**/pretrained_model/config.json"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(f"Could not find a LeRobot pretrained_model directory under {p}")


def resume_config(run_dir: Path) -> Path:
    cands = [
        run_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
        run_dir / "checkpoints" / "last" / "pretrained_model" / "config.json",
    ]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Cannot resume: no checkpoint config found under {run_dir}/checkpoints/last/pretrained_model")


def maybe_remove_output(run_dir: Path, *, overwrite: bool, resume: bool) -> None:
    if not run_dir.exists():
        return
    if resume:
        resume_config(run_dir)
        return
    if overwrite:
        shutil.rmtree(run_dir)
        return
    raise FileExistsError(
        f"Output directory {run_dir} already exists. Use --overwrite-output for a fresh run or --resume to continue."
    )


def train_entry(args, extra_lerobot_args: list[str]) -> int:
    env = runtime_env(args.data_dir, args.output_dir, args.cuda_id)
    os.environ.update(env)
    output_root = resolve_path(args.output_dir)
    mode = args.mode.upper()
    prepare_all(
        args.data_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force_download=args.force_download,
        force_prepare=args.force_prepare,
    )
    dataset_root = root_for_mode(args.data_dir, "ABC" if mode in {"ABC", "A,B,C"} else mode)
    run_dir = output_root / args.run_name
    maybe_remove_output(run_dir, overwrite=args.overwrite_output, resume=args.resume)

    device_arg = "--policy.device=cpu" if str(args.cuda_id).lower() in {"cpu", "none", "-1"} else "--policy.device=cuda"
    cmd = [
        require_lerobot_train(args.lerobot_train),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.revision={args.revision}",
        "--dataset.use_imagenet_stats=false",
        "--policy.type=act",
        device_arg,
        f"--output_dir={run_dir}",
        f"--job_name={args.run_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--seed={args.seed}",
        "--policy.push_to_hub=false",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
        f"--wandb.project={args.wandb_project}",
        f"--wandb.mode={args.wandb_mode}",
        "--wandb.disable_artifact=true",
    ]
    if int(args.eval_freq) > 0:
        cmd.append(f"--eval_freq={args.eval_freq}")
    if not args.use_lerobot_act_defaults:
        cmd.extend([
            "--policy.chunk_size=100",
            "--policy.n_action_steps=100",
            "--policy.vision_backbone=resnet18",
            "--policy.dim_model=512",
            "--policy.n_heads=8",
            "--policy.n_encoder_layers=4",
            "--policy.n_decoder_layers=1",
            "--policy.kl_weight=10.0",
            f"--policy.optimizer_lr={args.lr}",
            f"--policy.optimizer_weight_decay={args.weight_decay}",
        ])
    if args.resume:
        cmd.extend(["--resume=true", f"--config_path={resume_config(run_dir)}"])
    cmd.extend(extra_lerobot_args)

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "command": cmd,
        "hf_home": env.get("HF_HOME"),
        "hf_lerobot_home": env.get("HF_LEROBOT_HOME"),
    }
    write_json(output_root / "_metadata" / f"{args.run_name}.json", record)
    tqdm.write(f"[train] mode={mode}")
    tqdm.write(f"[train] dataset_root={dataset_root}")
    tqdm.write("[command]\n" + " ".join(shlex.quote(str(x)) for x in cmd))
    log_path = output_root / "_logs" / f"{args.run_name}_train.log"
    start_epoch = time.time()
    start_perf = time.perf_counter()
    returncode = run_command_with_log(cmd, env=env, log_path=log_path)
    elapsed_s = time.perf_counter() - start_perf
    status = "success" if returncode == 0 else "failed"

    metrics_row, metrics_raw = build_training_metrics_row(
        time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
        run_name=args.run_name,
        mode=mode,
        status=status,
        returncode=returncode,
        elapsed_s=elapsed_s,
        steps_requested=int(args.steps),
        batch_size=int(args.batch_size),
        dataset_root=dataset_root,
        run_dir=run_dir,
        log_path=log_path,
        output_dir=output_root,
        start_epoch=start_epoch,
    )
    write_training_metric_outputs(output_root, run_dir, metrics_row, metrics_raw)
    tqdm.write(f"[metrics] Saved per-run metrics to {run_dir / 'training_metrics.csv'}")
    tqdm.write(f"[metrics] Saved comparison table to {output_root / 'training_metrics_latest.csv'}")
    if "action_l1_loss" not in metrics_row:
        tqdm.write(
            "[metrics warning] action_l1_loss was not found in local wandb summary. "
            "The CSV still contains train_loss/grad_norm/lr parsed from the LeRobot log. "
            "For action_l1_loss, keep wandb enabled or use wandb offline mode so wandb-summary.json is written locally."
        )

    if returncode == 0:
        write_json(run_dir / "run_metadata.json", {**record, "training_metrics": metrics_row})
    return int(returncode)
