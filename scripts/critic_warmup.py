"""Phase 5b: critic warm-up on a BC-seeded policy.

Loads the BC model from the run dir, FREEZES the entire policy (trunk + action
head), and trains ONLY the value head on on-policy rollouts in the real training
distribution (epsilon curriculum + self-play). The policy cannot drift (frozen),
so this purely calibrates V(s) — directly attacking the diagnosed BC->PPO
collapse cause (garbage advantages from an uncalibrated critic). Then resets
num_timesteps to 0 and re-saves, so the subsequent RL run's LR-warmup applies to
the first real policy updates.

Pipeline: bc_pretrain.py -> critic_warmup.py -> orchestrator (train.py --resume).

Usage: uv run python scripts/critic_warmup.py [--config configs/bc_critic_warmup.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sb3_contrib import MaskablePPO  # noqa: E402

from catan_rl.training.ppo import _make_vec_env, _resolve_run_dir  # noqa: E402
from catan_rl.utils.config import load_yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bc_critic_warmup.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_dir = _resolve_run_dir(cfg)
    seed = run_dir / "model.zip"
    if not seed.exists():
        raise SystemExit(f"no BC seed at {seed}; run bc_pretrain.py first")
    marker = run_dir / "critic_warmup_done.txt"
    if marker.exists():
        print(f"[critic_warmup] already done ({marker}); skipping")
        return 0

    steps = cfg.train.critic_warmup_steps
    print(f"[critic_warmup] freezing policy, training value head for {steps:,} steps")

    venv = _make_vec_env(cfg, run_dir)
    model = MaskablePPO.load(str(seed), env=venv, device=cfg.train.device,
                             custom_objects={"verbose": 1})

    # Freeze everything except the value head.
    n_train = n_froze = 0
    for name, p in model.policy.named_parameters():
        train = name.startswith("value_net")
        p.requires_grad_(train)
        n_train += train
        n_froze += (not train)
    print(f"[critic_warmup] value-head params trainable: {n_train}, frozen: {n_froze}")

    # Value-head-only learning. No wandb (avoids step collisions with the RL run);
    # watch train/explained_variance climb in stdout — that's the calibration signal.
    model.learn(total_timesteps=steps, reset_num_timesteps=True, progress_bar=False)

    # Unfreeze for the RL phase, reset the step counter so LR-warmup is correct.
    for p in model.policy.parameters():
        p.requires_grad_(True)
    model.num_timesteps = 0
    try:
        model._num_timesteps_at_start = 0
    except Exception:
        pass

    model.save(str(seed))
    marker.write_text("done")
    print(f"[critic_warmup] DONE -> re-saved {seed} (num_timesteps reset to 0), marker written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
