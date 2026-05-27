"""CLI for Phase 1 training.

Usage:
    uv run python scripts/train.py --config configs/phase1.yaml
    uv run python scripts/train.py --config configs/phase1.yaml --resume
    uv run python scripts/train.py --config configs/phase1.yaml --total-timesteps 20000
"""
from __future__ import annotations

import argparse
import sys

from catan_rl.training.ppo import train
from catan_rl.utils.config import load_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--skip-final-ladder", action="store_true",
                   help="Skip the post-training 500-game ladder (saves ~10 min on AlphaBeta n=2).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml(args.config)
    if args.total_timesteps is not None:
        cfg.train.total_timesteps = args.total_timesteps
    if args.device is not None:
        cfg.train.device = args.device
    if args.n_envs is not None:
        cfg.train.n_envs = args.n_envs
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.wandb is not None:
        cfg.wandb.mode = args.wandb
    if args.skip_final_ladder:
        cfg.eval.run_final_ladder = False
    final = train(cfg, resume=args.resume)
    print(f"\n[train] done -> {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
