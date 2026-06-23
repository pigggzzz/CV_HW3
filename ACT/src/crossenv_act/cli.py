from __future__ import annotations

import argparse

from .constants import DEFAULT_PROJECT, DEFAULT_REPO_ID, DEFAULT_REVISION


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--cuda-id", default="0")
    p.add_argument("--force-download", action="store_true", help="Re-download raw data into <data-dir>/calvin_act_work/raw.")
    p.add_argument("--force-prepare", action="store_true", help="Rebuild prepared v3 ACT-compatible split roots.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossenv_act")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("inspect", help="Download, prepare and inspect CALVIN split roots.")
    add_common(prep)

    tr = sub.add_parser("train", help="Train LeRobot ACT on splitB or merged splitA/B/C.")
    add_common(tr)
    tr.add_argument("--mode", required=True, choices=["B", "ABC", "A,B,C"])
    tr.add_argument("--run-name", required=True)
    tr.add_argument("--steps", type=int, default=100000)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--log-freq", type=int, default=200)
    tr.add_argument("--save-freq", type=int, default=20000)
    tr.add_argument("--eval-freq", type=int, default=0)
    tr.add_argument("--seed", type=int, default=1000)
    tr.add_argument("--lr", default="1e-5")
    tr.add_argument("--weight-decay", default="1e-4")
    tr.add_argument("--wandb-enable", action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    tr.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--overwrite-output", action="store_true")
    tr.add_argument("--use-lerobot-act-defaults", action="store_true")
    tr.add_argument("--lerobot-train", default=None)

    ev = sub.add_parser("eval", help="Zero-shot offline action-error evaluation on splitD.")
    add_common(ev)
    ev.add_argument("--basic-policy", required=True)
    ev.add_argument("--joint-policy", required=True)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.add_argument("--max-batches", type=int, default=None, help="Smoke-test only. Omit this for full splitD zero-shot evaluation.")
    ev.add_argument("--max-samples", type=int, default=None)
    ev.add_argument("--tolerance-s", type=float, default=1e-4)
    ev.add_argument("--wandb-enable", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    ev.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.cmd == "inspect":
        from .inspect_cmd import inspect_entry
        return inspect_entry(args)
    if args.cmd == "train":
        from .train import train_entry
        return train_entry(args, extra)
    if args.cmd == "eval":
        from .eval import eval_entry
        return eval_entry(args)
    parser.error("Unknown command")
    return 2
