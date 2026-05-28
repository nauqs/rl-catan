"""Download W&B history for a run and produce PNG plots colocated with the model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402


def _plot(df, x: str, ys: list[tuple[str, str]], title: str, ylabel: str,
          out: Path, ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for y, label in ys:
        if y not in df.columns:
            continue
        d = df[[x, y]].dropna()
        if d.empty:
            continue
        ax.plot(d[x] / 1e6, d[y], label=label, alpha=0.85, linewidth=1.4)
    ax.set_xlabel("env steps (millions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="W&B run path, e.g. nauqs/catan-rl/uxqv42h2")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--samples", type=int, default=5000)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(args.run)
    print(f"run: {run.name} ({run.state}), steps: {run.summary.get('global_step', '?')}")

    df = run.history(samples=args.samples)
    df.to_csv(args.out_dir / "wandb_history.csv", index=False)
    x = "global_step"

    _plot(
        df, x,
        [("eval/1v1/random/winrate", "vs Random"),
         ("eval/1v1/weighted/winrate", "vs Weighted"),
         ("eval/1v1/value/winrate", "vs Value")],
        title="In-training winrate (deterministic, 20 games / snapshot)",
        ylabel="winrate", out=args.out_dir / "winrate.png",
        ylim=(-0.02, 1.02),
    )
    _plot(
        df, x,
        [("eval/1v1/random/mean_vp", "vs Random"),
         ("eval/1v1/weighted/mean_vp", "vs Weighted"),
         ("eval/1v1/value/mean_vp", "vs Value")],
        title="In-training mean VP",
        ylabel="mean VP", out=args.out_dir / "mean_vp.png",
    )
    _plot(
        df, x,
        [("rollout/ep_rew_mean", "ep_rew_mean")],
        title="Mean episode reward (training rollouts)",
        ylabel="reward", out=args.out_dir / "ep_rew_mean.png",
    )
    _plot(
        df, x,
        [("train/value_loss", "value_loss"),
         ("train/entropy_loss", "entropy_loss"),
         ("train/policy_gradient_loss", "policy_gradient_loss")],
        title="Loss components",
        ylabel="value", out=args.out_dir / "losses.png",
    )
    _plot(
        df, x,
        [("train/explained_variance", "explained_variance"),
         ("train/approx_kl", "approx_kl"),
         ("train/clip_fraction", "clip_fraction")],
        title="PPO diagnostics",
        ylabel="value", out=args.out_dir / "ppo_diagnostics.png",
    )
    _plot(
        df, x,
        [("time/fps", "fps")],
        title="Throughput (env steps / second)",
        ylabel="fps", out=args.out_dir / "fps.png",
    )

    print(f"wrote plots + wandb_history.csv to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
