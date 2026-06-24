"""AlphaBeta-ExIt: improve the value net so it beats the handcrafted base_fn at EQUAL depth.

Fairness: AB2 = base_fn value + depth-2 search. Comparing our depth-3
search to it is outsearching, not a better value. The real goal is a LEARNED value that,
at the SAME depth, beats base_fn -> NeuralAB(depth2) vs AB2(depth2) > 0.5.

Loop (one round = generate -> train value -> eval):
  1. Generate self-play games with NeuralAlphaBetaPlayer(policy, value, gen_depth, topk).
     Record (board, numeric, dt, won) at each player's decision -- i.e. on the states the
     SEARCH actually visits (the right training distribution; value-vs-value overfits, as
     we found: more value-vs-value EV -> worse search).
  2. Retrain the value head (frozen policy) on those (state, discounted-outcome) pairs.
  3. Eval the FAIR metric: NeuralAB(eval_depth=2, new value) vs AB2(depth2), plus a
     base_fn+our-pruning control (isolates value head vs base_fn under identical search).
  Iterate: better value -> stronger search -> regenerate -> ...

Usage:
  python scripts/exit_alphabeta.py --rounds 3 --games 600 --gen-depth 2 --topk 3 \
      --epochs 12 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

GEN_PER_TASK = 50


# --------------------------------------------------------------------------- #
# Stage 1: parallel self-play generation (records states the SEARCH visits)
# --------------------------------------------------------------------------- #

def _gen_shard(args):
    seed_start, n_games, out_path, policy_path, value_path, depth, topk = args
    import numpy as np
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catan_rl.search.neural_alphabeta import NeuralAlphaBetaPlayer

    class _RecAB(NeuralAlphaBetaPlayer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.rows = []  # (board f16, numeric f32, turn)
        def decide(self, game, playable):
            if len(playable) > 1:
                obs = self._rl._build_obs(game, self.color)
                self.rows.append((obs["board"].astype(np.float16), obs["numeric"]))
            return super().decide(game, playable)

    boards, numerics, dts, wons = [], [], [], []
    # Load the models ONCE per worker (not per game) — model inference is stateless.
    blue = _RecAB(Color.BLUE, policy_path, depth=depth, policy_topk=topk, device="cpu",
                  value_model_path=value_path)
    red = _RecAB(Color.RED, policy_path, depth=depth, policy_topk=topk, device="cpu",
                 value_model_path=value_path)
    for s in range(seed_start, seed_start + n_games):
        blue.rows = []; red.rows = []
        g = Game([blue, red], seed=s)
        g.play()
        won_blue = int(g.winning_color() == Color.BLUE)
        for player, won in ((blue, won_blue), (red, 1 - won_blue)):
            n = len(player.rows)
            for i, (b, nu) in enumerate(player.rows):
                boards.append(b); numerics.append(nu)
                dts.append(n - i); wons.append(won)
    np.savez_compressed(
        out_path,
        boards=np.stack(boards), numerics=np.stack(numerics),
        dts=np.array(dts, dtype=np.int32), wons=np.array(wons, dtype=np.int8),
    )
    return out_path, len(dts)


def generate(n_games, policy_path, value_path, depth, topk, workers, shard_dir):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks, s, i = [], 0, 0
    while s < n_games:
        cnt = min(GEN_PER_TASK, n_games - s)
        tasks.append((s, cnt, str(shard_dir / f"shard_{i:03d}.npz"),
                      str(policy_path), str(value_path) if value_path else None, depth, topk))
        s += cnt; i += 1
    t0 = time.time(); paths = []; total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, n in ex.map(_gen_shard, tasks):
            paths.append(Path(p)); total += n
            print(f"  shard {p}: +{n} (total {total}, {(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"GEN: {total} positions from {n_games} games in {(time.time()-t0)/60:.1f}min", flush=True)
    return paths


# --------------------------------------------------------------------------- #
# Stage 2: value-head training on self-play outcomes (frozen policy)
# --------------------------------------------------------------------------- #

def train_value(seed_model, shards, out, epochs, lr, batch, gamma):
    import torch as th
    from sb3_contrib import MaskablePPO
    from catan_rl.players.rl_player import _force_cpu_deserialize

    data = [np.load(p) for p in shards]
    boards = np.concatenate([d["boards"] for d in data])
    numerics = np.concatenate([d["numerics"] for d in data])
    dts = np.concatenate([d["dts"] for d in data])
    wons = np.concatenate([d["wons"] for d in data])
    del data
    n = len(dts)
    vtarget = (gamma ** dts) * np.where(wons == 1, 500.0, -500.0)
    print(f"  value dataset: {n} positions", flush=True)

    with _force_cpu_deserialize():
        model = MaskablePPO.load(str(seed_model), device="mps")
    pol = model.policy
    pol.set_training_mode(True)
    for name, p in pol.named_parameters():
        p.requires_grad_(name.startswith("value_net"))
    vparams = [p for nm, p in pol.named_parameters() if nm.startswith("value_net")]
    opt = th.optim.Adam(vparams, lr=lr)
    dev = model.device

    rng = np.random.default_rng(0)
    holdout = max(1, int(0.03 * n)); perm0 = rng.permutation(n)
    val_idx, train_idx = perm0[:holdout], perm0[holdout:]

    def run_batch(idx, train):
        obs = {"board": th.tensor(boards[idx], dtype=th.float32, device=dev),
               "numeric": th.tensor(numerics[idx], dtype=th.float32, device=dev)}
        vt = th.tensor(vtarget[idx], dtype=th.float32, device=dev)
        feats = pol.extract_features(obs)
        fm, g = pol._split(feats)
        v = pol._value(fm, g).squeeze(-1)
        loss = th.nn.functional.mse_loss(v / 500.0, vt / 500.0)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return v.detach().cpu().numpy(), vt.detach().cpu().numpy()

    for ep in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            run_batch(order[i:i + batch], True)
        vs, vts = [], []
        for i in range(0, len(val_idx), batch):
            vp, vtt = run_batch(val_idx[i:i + batch], False)
            vs.append(vp); vts.append(vtt)
        vs = np.concatenate(vs); vts = np.concatenate(vts)
        ev = 1 - np.var(vts - vs) / (np.var(vts) + 1e-8)
        print(f"  ep {ep}: EV {ev:.3f}", flush=True)
    model.save(str(out))
    print(f"  saved {out}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 3: fair eval — NeuralAB(depth2) vs AB2(depth2)
# --------------------------------------------------------------------------- #

def eval_fair(policy_path, value_path, depth, topk, games, seed0=900):
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catan_rl.search.neural_alphabeta import NeuralAlphaBetaPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    w = 0
    for s in range(games):
        agent = NeuralAlphaBetaPlayer(Color.BLUE, policy_path, depth=depth, policy_topk=topk,
                                      device="cpu", value_model_path=value_path)
        g = Game([agent, AlphaBetaPlayer(Color.RED, depth=2)], seed=seed0 + s)
        g.play()
        w += g.winning_color() == Color.BLUE
    return w / games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/bc_seed/model.zip")
    ap.add_argument("--value", default="checkpoints/bc_seed/model.zip")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--gen-depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-games", type=int, default=30)
    ap.add_argument("--out-dir", default="checkpoints/exit_ab")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    value = args.value
    print(f"[exit] fair baseline: NeuralAB(d2,topk{args.topk}) vs AB2(d2), round-0 value=seed", flush=True)
    wr0 = eval_fair(args.policy, value, 2, args.topk, args.eval_games)
    print(f"[exit] ROUND 0 (seed value): depth2 vs AB2 = {wr0:.3f}", flush=True)

    for r in range(1, args.rounds + 1):
        print(f"\n===== ROUND {r} =====", flush=True)
        shard_dir = out_dir / f"shards_r{r}"
        shards = generate(args.games, args.policy, value, args.gen_depth, args.topk,
                          args.workers, shard_dir)
        new_value = out_dir / f"value_r{r}.zip"
        train_value(args.policy if value == args.policy else value, shards, new_value,
                    args.epochs, args.lr, args.batch, args.gamma)
        value = str(new_value)
        wr = eval_fair(args.policy, value, 2, args.topk, args.eval_games)
        print(f"[exit] ROUND {r}: depth2 vs AB2 = {wr:.3f}  (target >0.5)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
