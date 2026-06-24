"""AlphaBeta-ExIt, POLICY half: distill the search's move choices into the policy.

The value-MSE half failed (outcome regression -> high EV but poor move-ranker, worse
search, twice). The strength actually lives in the POLICY + search depth, so improve the
POLICY: self-play with the searcher, record the move the search PLAYS, train the policy
to imitate it (masked CE, like BC but the teacher is the search, which plays better than
the raw policy). Better policy -> better pruning + better greedy -> stronger search -> iterate
(this is AlphaZero's actual strength driver).

Decoupling: the policy net (distilled) and the value net are SEPARATE checkpoints.
NeuralAlphaBetaPlayer uses policy_model for pruning + a fixed `value_model` (original BC
value) for leaf eval -- so distilling the policy's trunk can't corrupt the value.

Eval (fair, equal depth): NeuralAB(d2, distilled policy, orig value) vs AB2(d2) > 0.5,
plus the distilled policy GREEDY vs the ladder (does the policy itself get stronger?).

Usage: python scripts/exit_policy.py --rounds 4 --games 600 --gen-depth 2 --topk 3 \
       --epochs 6 --workers 6
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


def _gen_shard(args):
    seed_start, n_games, out_path, policy_path, value_path, depth, topk = args
    import numpy as np
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catanatron.gym.envs.action_space import get_action_array, to_action_space
    from catan_rl.search.neural_alphabeta import NeuralAlphaBetaPlayer

    class _RecAB(NeuralAlphaBetaPlayer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.rows = []  # (board f16, numeric f32, mask bool, action_idx)
        def decide(self, game, playable):
            action = super().decide(game, playable)
            if len(playable) > 1:
                obs = self._rl._build_obs(game, self.color)
                pcs = tuple(p.color for p in game.state.players)
                mask = self._rl._build_mask(pcs, playable)
                aidx = to_action_space(action, pcs, self._rl._map_type)
                self.rows.append((obs["board"].astype(np.float16), obs["numeric"], mask, aidx))
            return action

    boards, numerics, masks, actions = [], [], [], []
    blue = _RecAB(Color.BLUE, policy_path, depth=depth, policy_topk=topk, device="cpu",
                  value_model_path=value_path)
    red = _RecAB(Color.RED, policy_path, depth=depth, policy_topk=topk, device="cpu",
                 value_model_path=value_path)
    for s in range(seed_start, seed_start + n_games):
        blue.rows = []; red.rows = []
        g = Game([blue, red], seed=s)
        g.play()
        for player in (blue, red):
            for b, nu, m, a in player.rows:
                boards.append(b); numerics.append(nu); masks.append(m); actions.append(a)
    np.savez_compressed(
        out_path,
        boards=np.stack(boards), numerics=np.stack(numerics),
        masks=np.stack(masks), actions=np.array(actions, dtype=np.int32),
    )
    return out_path, len(actions)


def generate(n_games, policy_path, value_path, depth, topk, workers, shard_dir):
    shard_dir = Path(shard_dir); shard_dir.mkdir(parents=True, exist_ok=True)
    tasks, s, i = [], 0, 0
    while s < n_games:
        cnt = min(GEN_PER_TASK, n_games - s)
        tasks.append((s, cnt, str(shard_dir / f"shard_{i:03d}.npz"),
                      str(policy_path), str(value_path), depth, topk))
        s += cnt; i += 1
    t0 = time.time(); paths = []; total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, n in ex.map(_gen_shard, tasks):
            paths.append(Path(p)); total += n
            print(f"  shard {p}: +{n} (total {total}, {(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"GEN: {total} positions from {n_games} games in {(time.time()-t0)/60:.1f}min", flush=True)
    return paths


def train_policy(seed_model, shards, out, epochs, lr, batch):
    import torch as th
    from sb3_contrib import MaskablePPO
    from catan_rl.players.rl_player import _force_cpu_deserialize

    data = [np.load(p) for p in shards]
    boards = np.concatenate([d["boards"] for d in data])
    numerics = np.concatenate([d["numerics"] for d in data])
    masks = np.concatenate([d["masks"] for d in data])
    actions = np.concatenate([d["actions"] for d in data])
    del data
    n = len(actions)
    print(f"  policy dataset: {n} (state, search-move) pairs", flush=True)

    with _force_cpu_deserialize():
        model = MaskablePPO.load(str(seed_model), device="mps")
    pol = model.policy
    pol.set_training_mode(True)
    for name, p in pol.named_parameters():
        p.requires_grad_(not name.startswith("value_net"))  # train trunk + action head
    tparams = [p for nm, p in pol.named_parameters() if not nm.startswith("value_net")]
    opt = th.optim.Adam(tparams, lr=lr)
    dev = model.device

    rng = np.random.default_rng(0)
    holdout = max(1, int(0.03 * n)); perm0 = rng.permutation(n)
    val_idx, train_idx = perm0[:holdout], perm0[holdout:]

    def run_batch(idx, train):
        obs = {"board": th.tensor(boards[idx], dtype=th.float32, device=dev),
               "numeric": th.tensor(numerics[idx], dtype=th.float32, device=dev)}
        m = th.tensor(masks[idx], device=dev)
        y = th.tensor(actions[idx], dtype=th.long, device=dev)
        feats = pol.extract_features(obs)
        fm, g = pol._split(feats)
        logits = pol.action_head(fm, g).masked_fill(~m, -1e8)
        ce = th.nn.functional.cross_entropy(logits, y)
        if train:
            opt.zero_grad(); ce.backward(); opt.step()
        acc = float((logits.argmax(1) == y).float().mean())
        return float(ce), acc

    for ep in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            run_batch(order[i:i + batch], True)
        vces, vaccs = [], []
        for i in range(0, len(val_idx), batch):
            ce, acc = run_batch(val_idx[i:i + batch], False)
            vces.append(ce); vaccs.append(acc)
        print(f"  ep {ep}: val_ce {np.mean(vces):.3f} val_acc {np.mean(vaccs):.3f}", flush=True)
    model.save(str(out))
    print(f"  saved {out}", flush=True)


def eval_vs_ab2(policy_path, value_path, depth, topk, games, seed0=900):
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


def eval_greedy(policy_path, opp, games, seed0=900):
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catan_rl.players.rl_player import RLPlayer
    from catan_rl.eval.opponents import build_opponent
    w = 0
    for s in range(games):
        agent = RLPlayer(Color.BLUE, Path(policy_path), representation="mixed",
                         enable_maritime_trade=True, deterministic=True)
        g = Game([agent, build_opponent(opp, Color.RED)], seed=seed0 + s)
        g.play()
        w += g.winning_color() == Color.BLUE
    return w / games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/bc_seed/model.zip")
    ap.add_argument("--value", default="checkpoints/bc_seed/model.zip")  # FIXED original BC value
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--gen-depth", type=int, default=2)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-games", type=int, default=30)
    ap.add_argument("--out-dir", default="checkpoints/exit_pol")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    policy = args.policy
    g_value = eval_greedy(policy, "alphabeta2", args.eval_games)
    s_value = eval_vs_ab2(policy, args.value, 2, args.topk, args.eval_games)
    print(f"[exitP] ROUND 0: policy-greedy vs AB2 = {g_value:.3f} ; search(d2) vs AB2 = {s_value:.3f}", flush=True)

    for r in range(1, args.rounds + 1):
        print(f"\n===== ROUND {r} =====", flush=True)
        shards = generate(args.games, policy, args.value, args.gen_depth, args.topk,
                          args.workers, out_dir / f"shards_r{r}")
        new_policy = out_dir / f"policy_r{r}.zip"
        train_policy(policy, shards, new_policy, args.epochs, args.lr, args.batch)
        policy = str(new_policy)
        g_v = eval_greedy(policy, "alphabeta2", args.eval_games)
        s_v = eval_vs_ab2(policy, args.value, 2, args.topk, args.eval_games)
        print(f"[exitP] ROUND {r}: policy-greedy vs AB2 = {g_v:.3f} ; search(d2) vs AB2 = {s_v:.3f}  (target >0.5)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
