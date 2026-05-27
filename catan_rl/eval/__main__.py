"""CLI: `python -m catan_rl.eval ...`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catanatron import Color, RandomPlayer

from catan_rl.eval.ladder import run_ladder
from catan_rl.eval.opponents import OPPONENTS


def _candidate_factory(spec: str):
    if spec == "random":
        return lambda color: RandomPlayer(color)
    if spec.startswith("checkpoint:"):
        from catan_rl.players.rl_player import RLPlayer

        path = spec.split(":", 1)[1]
        return lambda color: RLPlayer(color, path)
    raise SystemExit(f"unknown candidate '{spec}' (expected 'random' or 'checkpoint:<path>')")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="catan_rl.eval")
    p.add_argument("--candidate", required=True, help="'random' or 'checkpoint:<path>'")
    p.add_argument(
        "--opponent",
        default="all",
        help=f"opponent name or 'all'. known: {','.join(sorted(OPPONENTS))}",
    )
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--mode", choices=["1v1", "4p", "both"], default="both")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    p.add_argument("--wandb", choices=["online", "offline", "disabled"], default="disabled")
    p.add_argument("--wandb-project", default="catan-rl")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    candidate = _candidate_factory(args.candidate)
    opponents = list(OPPONENTS) if args.opponent == "all" else [args.opponent]
    modes = ("1v1", "4p") if args.mode == "both" else (args.mode,)

    run = None
    if args.wandb != "disabled":
        from catan_rl.utils.wandb import init_wandb

        run = init_wandb(
            project=args.wandb_project,
            mode=args.wandb,
            config={
                "candidate": args.candidate,
                "opponents": opponents,
                "modes": list(modes),
                "games": args.games,
            },
        )

    result = run_ladder(
        candidate,
        candidate_name=args.candidate,
        opponents=opponents,
        n_games=args.games,
        modes=modes,
        base_seed=args.seed,
    )

    payload = result.to_dict()
    print(json.dumps(payload, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    if run is not None:
        import wandb

        for row in payload["results"]:
            wandb.log(
                {
                    f"eval/{row['mode']}/{row['opponent']}/winrate": row["winrate"],
                    f"eval/{row['mode']}/{row['opponent']}/mean_vp": row["mean_vp"],
                    f"eval/{row['mode']}/{row['opponent']}/mean_turns": row["mean_turns"],
                }
            )
        wandb.summary["results"] = payload["results"]
        run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
