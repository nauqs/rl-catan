"""Expert Iteration (ExIt / AlphaZero-style), PPO-free.

The wall this project keeps hitting is that BC-then-PPO can preserve the teacher
(value bot) but never *exceed* it: PPO chases the value-net proxy, which doesn't
equal winning, and drifts/collapses. ExIt breaks past the teacher because the
training target comes from *search*, which is strictly stronger than the raw net:

  expert (slow):  UCB1 root search guided by the current net (SearchRLPlayer)
  apprentice:     the net, trained by SUPERVISED imitation of the expert ->
                  policy target = search visit-count distribution,
                  value target  = game outcome.
  iterate:        better net -> better-guided search -> better targets -> ...

No policy gradient, no advantages, no PPO — so none of the PPO instability. Each
round is self-play generation (search both sides, temperature sampling for
exploration) followed by a supervised distillation pass, starting from the BC
seed and improving on it.

Usage:
  python scripts/exit_train.py --config configs/exit.yaml \
      --rounds 10 --games-per-round 300 --n-sims 64 --depth 8 \
      --temperature 1.0 --workers 5
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.bc_pretrain as bp  # reuse build_model + config plumbing
from catan_rl.utils.config import load_yaml

VF_COEF = 0.3
EPOCHS = 4
BATCH = 512


# --------------------------------------------------------------------------- #
# Generation: UCB1 self-play, record visit-count targets
# --------------------------------------------------------------------------- #

def _gen_shard_exit(args):
    (seed_start, n_games, out_path, net_path, n_sims, depth, temperature,
     opponent) = args
    from catanatron.game import Game
    from catanatron.models.player import Color, Player
    from catanatron.features import create_sample, get_feature_ordering
    from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
    from catanatron.gym.envs.action_space import get_action_array, to_action_space
    from catan_rl.search.ucb_player import SearchRLPlayer
    from catan_rl.eval.opponents import build_opponent

    feats = get_feature_ordering(2, "BASE")
    numeric_feats = [f for f in feats if not is_graph_feature(f)]

    def make_searcher(color):
        return SearchRLPlayer(
            color, net_path, n_sims=n_sims, rollout_depth=depth,
            rollout_policy="policy", policy_opponent=True,
            temperature=temperature, representation="mixed",
            enable_maritime_trade=True, seed=seed_start + 1,
        )

    boards, numerics, masks, actions, dts, wons = [], [], [], [], [], []

    class Recorder(Player):
        """Wraps a SearchRLPlayer; records (obs, search's chosen move).

        Target = the move the search PICKS (argmax mean-Q), recorded as a hard
        label — NOT the visit-count distribution, which is ~uniform at low
        n_sims (every root arm gets ~1 sim) and distills the policy toward
        random. argmax-Q is the same expert signal that amplified the raw net
        in the Phase-4 ablation, so cloning it lifts the net above the teacher.
        """

        def __init__(self, color):
            super().__init__(color)
            self.inner = make_searcher(color)
            self.rows = []

        def decide(self, game, playable):
            action = self.inner.decide(game, playable)
            la = self.inner.last_actions
            if la is not None and len(la) > 1:
                pcs = tuple(p.color for p in game.state.players)
                board = create_board_tensor(game, self.color, channels_first=True)
                sample = create_sample(game, self.color)
                numeric = np.array([sample[f] for f in numeric_feats], dtype=np.float32)
                arr = get_action_array(pcs, "BASE")
                mask = np.zeros(len(arr), dtype=bool)
                for a in la:
                    mask[to_action_space(a, pcs, "BASE")] = True
                aidx = to_action_space(action, pcs, "BASE")
                self.rows.append((board.astype(np.float16), numeric, mask, aidx))
            return action

    for s in range(seed_start, seed_start + n_games):
        blue = Recorder(Color.BLUE)
        if opponent == "self":
            red = Recorder(Color.RED)
            players = [blue, red]
            recorders = [blue, red]
        else:
            red = build_opponent(opponent, Color.RED)
            players = [blue, red]
            recorders = [blue]
        g = Game(players, seed=s)
        g.play()
        won_blue = int(g.winning_color() == Color.BLUE)
        for rec in recorders:
            won = won_blue if rec.color == Color.BLUE else (1 - won_blue)
            nrows = len(rec.rows)
            for i, (b, nu, m, a) in enumerate(rec.rows):
                boards.append(b); numerics.append(nu); masks.append(m)
                actions.append(a); dts.append(nrows - i); wons.append(won)

    # Same schema as bc_pretrain shards, so distillation reuses bc_train().
    np.savez_compressed(
        out_path,
        boards=np.stack(boards), numerics=np.stack(numerics),
        masks=np.stack(masks), actions=np.array(actions, dtype=np.int32),
        dts=np.array(dts, dtype=np.int32), wons=np.array(wons, dtype=np.int8),
    )
    return out_path, len(actions)


def generate_round(net_path, shard_dir, n_games, n_sims, depth, temperature,
                   opponent, workers):
    shard_dir.mkdir(parents=True, exist_ok=True)
    per = max(1, n_games // max(1, workers))
    tasks, s, i = [], 0, 0
    while s < n_games:
        cnt = min(per, n_games - s)
        tasks.append((s, cnt, str(shard_dir / f"shard_{i:03d}.npz"),
                      str(net_path), n_sims, depth, temperature, opponent))
        s += cnt; i += 1
    t0 = time.time()
    paths, total = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, n in ex.map(_gen_shard_exit, tasks):
            paths.append(Path(p)); total += n
            print(f"    shard {Path(p).name}: +{n} (total {total}, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"  GEN: {total} decisions / {n_games} games in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return paths


# --------------------------------------------------------------------------- #
# Distillation reuses bc_pretrain.bc_train (hard cross-entropy + value MSE).
# --------------------------------------------------------------------------- #

def evaluate(net_path, games=20):
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catan_rl.players.rl_player import RLPlayer
    from catan_rl.eval.opponents import build_opponent
    out = {}
    for opp in ("value", "alphabeta1", "alphabeta2"):
        w = 0
        for s in range(games):
            g = Game([RLPlayer(Color.BLUE, net_path, representation="mixed",
                               enable_maritime_trade=True, deterministic=True),
                      build_opponent(opp, Color.RED)], seed=4000 + s)
            g.play()
            w += g.winning_color() == Color.BLUE
        out[opp] = w / games
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exit.yaml")
    ap.add_argument("--seed-model", default=None,
                    help="Initial net (default: run_dir/model.zip, else BC seed).")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--games-per-round", type=int, default=300)
    ap.add_argument("--n-sims", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = play argmax-Q (matches the recorded target). >0 "
                         "samples plays from visit counts (exploration), but the "
                         "target stays the search's argmax-Q move.")
    ap.add_argument("--opponent", default="self", help="'self' (search self-play) or a bot name")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=20)
    args = ap.parse_args()

    bp.CONFIG = Path(args.config)
    cfg = load_yaml(str(bp.CONFIG))
    run_dir = ROOT / "checkpoints" / "phase2" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    net_path = run_dir / "model.zip"

    if not net_path.exists():
        seed = args.seed_model or str(
            ROOT / "checkpoints/phase2/kl-anchor-adaptive/bc_ref.zip")
        import shutil
        shutil.copyfile(seed, net_path)
        print(f"seeded {net_path} from {seed}", flush=True)

    from catan_rl.utils.keep_awake import keep_awake
    keep_awake()

    e0 = evaluate(str(net_path), args.eval_games)
    print(f"[exit] round -1 (seed) eval: {e0}", flush=True)

    for rnd in range(args.rounds):
        print(f"\n[exit] === round {rnd} ===", flush=True)
        shard_dir = Path(f"/tmp/exit_shards/round_{rnd}")
        paths = generate_round(net_path, shard_dir, args.games_per_round,
                               args.n_sims, args.depth, args.temperature,
                               args.opponent, args.workers)
        # Load the current net and clone the search's chosen moves into it
        # (hard-target distillation = bc_train, teacher = our own search).
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(net_path), device=cfg.train.device)
        model = bp.bc_train(model, cfg, paths)
        model.save(str(net_path))
        model.save(str(run_dir / "self_snapshot.zip"))
        ev = evaluate(str(net_path), args.eval_games)
        print(f"[exit] round {rnd} eval: {ev}", flush=True)
        (run_dir / "exit_progress.txt").write_text(
            f"round {rnd}: {ev}\n", )

    print("[exit] DONE", flush=True)


if __name__ == "__main__":
    main()
