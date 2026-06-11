"""Phase 5 stage 1: behavior-clone ValueFunctionPlayer, seed the RL run dir.

  1. Generate value-vs-value games in parallel; record (board f16, numeric,
     action-mask, expert action, steps-to-end, won) for the BLUE side.
  2. Build a MaskablePPO with the EXACT phase5 hypers (so `train.py --resume`
     inherits them), BC-train the policy with masked cross-entropy and pretrain
     the value head on discounted terminal outcomes (so PPO's first advantages
     aren't garbage).
  3. Eval vs value/weighted/alphabeta1; seed run dir: model.zip,
     self_snapshot.zip (self-play starts at BC strength), epsilon.txt.

Usage: uv run python scripts/bc_pretrain.py [n_games=8000]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

CONFIG = ROOT / "configs" / "bc_ppo.yaml"  # overridden in main()
SHARD_DIR = Path("/tmp/bc_shards")
GAMES_PER_TASK = 250
EPOCHS = 6
BATCH = 512
VF_COEF = 0.3
EVAL_GAMES = 30


# ---------------------------------------------------------------------------
# Stage 1: parallel demonstration generation (workers write npz shards)
# ---------------------------------------------------------------------------

def _make_teacher(name, color):
    if name == "value":
        from catanatron.players.value import ValueFunctionPlayer
        return ValueFunctionPlayer(color)
    if name == "alphabeta1":
        from catanatron.players.minimax import AlphaBetaPlayer
        return AlphaBetaPlayer(color, depth=1)
    if name == "alphabeta2":
        from catanatron.players.minimax import AlphaBetaPlayer
        return AlphaBetaPlayer(color, depth=2)
    raise ValueError(name)


def _gen_shard(args):
    seed_start, n_games, out_path, teacher = args
    from catanatron.game import Game
    from catanatron.models.player import Color, Player
    from catanatron.features import create_sample, get_feature_ordering
    from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
    from catanatron.gym.envs.action_space import get_action_array, to_action_space

    feats = get_feature_ordering(2, "BASE")
    numeric_feats = [f for f in feats if not is_graph_feature(f)]

    boards, numerics, masks, actions, dts, wons = [], [], [], [], [], []

    class Recorder(Player):
        """Delegates decisions to the teacher bot, recording (obs, action)."""

        def __init__(self, color):
            super().__init__(color)
            self.inner = _make_teacher(teacher, color)
            self.rows = []

        def decide(self, game, playable):
            action = self.inner.decide(game, playable)
            if len(playable) > 1:
                pcs = tuple(p.color for p in game.state.players)
                board = create_board_tensor(game, self.color, channels_first=True)
                sample = create_sample(game, self.color)
                numeric = np.array([sample[f] for f in numeric_feats], dtype=np.float32)
                arr = get_action_array(pcs, "BASE")
                mask = np.zeros(len(arr), dtype=bool)
                for a in playable:
                    mask[to_action_space(a, pcs, "BASE")] = True
                aidx = to_action_space(action, pcs, "BASE")
                self.rows.append((board.astype(np.float16), numeric, mask, aidx))
            return action

    for s in range(seed_start, seed_start + n_games):
        rec = Recorder(Color.BLUE)
        g = Game([rec, _make_teacher(teacher, Color.RED)], seed=s)
        g.play()
        won = int(g.winning_color() == Color.BLUE)
        n = len(rec.rows)
        for i, (b, nu, m, a) in enumerate(rec.rows):
            boards.append(b); numerics.append(nu); masks.append(m)
            actions.append(a); dts.append(n - i); wons.append(won)

    np.savez_compressed(
        out_path,
        boards=np.stack(boards), numerics=np.stack(numerics),
        masks=np.stack(masks), actions=np.array(actions, dtype=np.int32),
        dts=np.array(dts, dtype=np.int32), wons=np.array(wons, dtype=np.int8),
    )
    return out_path, len(actions)


def generate(n_games: int, teacher: str = "value", workers: int = 6) -> list[Path]:
    shard_dir = SHARD_DIR if teacher == "value" else Path(f"/tmp/bc_shards_{teacher}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    s = 0
    i = 0
    while s < n_games:
        cnt = min(GAMES_PER_TASK, n_games - s)
        tasks.append((s, cnt, str(shard_dir / f"shard_{i:03d}.npz"), teacher))
        s += cnt; i += 1
    # Reuse existing shards (deterministic seeds) to skip regeneration.
    if all(Path(t[2]).exists() for t in tasks):
        print(f"  reusing {len(tasks)} existing shards in {shard_dir}", flush=True)
        return [Path(t[2]) for t in tasks]
    t0 = time.time()
    paths = []
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, n in ex.map(_gen_shard, tasks):
            paths.append(Path(p)); total += n
            print(f"  shard {p}: +{n} (total {total}, {time.time()-t0:.0f}s)", flush=True)
    print(f"GEN DONE: {total} decisions from {n_games} games in {(time.time()-t0)/60:.1f} min", flush=True)
    return paths


# ---------------------------------------------------------------------------
# Stage 2: BC training on the phase5-hyper model
# ---------------------------------------------------------------------------

def build_model(trunk_kwargs: dict | None = None, value_head: str = "mlp"):
    """MaskablePPO constructed exactly as ppo.train() would for phase5."""
    import gymnasium as gym
    import catanatron.gym  # noqa
    from sb3_contrib import MaskablePPO
    from catanatron.models.player import Color
    from catan_rl.features import HexFormer
    from catan_rl.policies.spatial import SpatialMaskableActorCriticPolicy
    from catan_rl.utils.config import load_yaml

    cfg = load_yaml(str(CONFIG))
    env = gym.make("catanatron/Catanatron-v0",
                   config={"map_type": "BASE", "representation": "mixed"})

    lr_peak = cfg.train.ppo.learning_rate
    warmup = cfg.train.ppo.warmup_steps
    total = cfg.train.total_timesteps
    if warmup and warmup > 0:
        def learning_rate(progress_remaining: float) -> float:
            step = total * (1.0 - progress_remaining)
            return lr_peak * min(1.0, step / warmup)
    else:
        learning_rate = lr_peak

    model = MaskablePPO(
        SpatialMaskableActorCriticPolicy, env,
        gamma=cfg.train.ppo.gamma, gae_lambda=cfg.train.ppo.gae_lambda,
        n_steps=cfg.train.ppo.n_steps, batch_size=cfg.train.ppo.batch_size,
        ent_coef=cfg.train.ppo.ent_coef, learning_rate=learning_rate,
        n_epochs=cfg.train.ppo.n_epochs, clip_range=cfg.train.ppo.clip_range,
        vf_coef=cfg.train.ppo.vf_coef, max_grad_norm=cfg.train.ppo.max_grad_norm,
        policy_kwargs=dict(
            features_extractor_class=HexFormer,
            features_extractor_kwargs=(trunk_kwargs or {}),
            player_colors=(Color.BLUE, Color.RED),
            map_type="BASE", enable_maritime=True,
            value_head=value_head,
        ),
        device="mps", verbose=0,
    )
    return model, cfg


def bc_train(model, cfg, shards: list[Path]):
    import torch as th
    dev = model.device
    data = [np.load(p) for p in shards]
    boards = np.concatenate([d["boards"] for d in data])
    numerics = np.concatenate([d["numerics"] for d in data])
    masks = np.concatenate([d["masks"] for d in data])
    actions = np.concatenate([d["actions"] for d in data])
    dts = np.concatenate([d["dts"] for d in data])
    wons = np.concatenate([d["wons"] for d in data])
    del data
    n = len(actions)
    print(f"BC dataset: {n} decisions ({boards.nbytes/1e9:.1f} GB boards f16)", flush=True)

    gamma = cfg.train.ppo.gamma
    win_r, loss_r = cfg.train.reward.win_reward, cfg.train.reward.loss_reward
    vtarget = (gamma ** dts) * np.where(wons == 1, win_r, loss_r)

    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)
    holdout = max(1, int(0.03 * n))
    rng = np.random.default_rng(0)
    perm0 = rng.permutation(n)
    val_idx, train_idx = perm0[:holdout], perm0[holdout:]

    def batch_loss(idx, train: bool):
        obs = {"board": th.tensor(boards[idx], dtype=th.float32, device=dev),
               "numeric": th.tensor(numerics[idx], dtype=th.float32, device=dev)}
        m = th.tensor(masks[idx], device=dev)
        y = th.tensor(actions[idx], dtype=th.long, device=dev)
        vt = th.tensor(vtarget[idx], dtype=th.float32, device=dev)
        feats = pol.extract_features(obs)
        fm, g = pol._split(feats)
        logits = pol.action_head(fm, g).masked_fill(~m, -1e8)
        v = pol._value(fm, g).squeeze(-1)
        ce = th.nn.functional.cross_entropy(logits, y)
        vmse = th.nn.functional.mse_loss(v / 500.0, vt / 500.0)
        loss = ce + VF_COEF * vmse
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        acc = float((logits.argmax(1) == y).float().mean())
        return float(ce), float(vmse), acc

    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(train_idx)
        ces, accs = [], []
        for i in range(0, len(order), BATCH):
            ce, vm, acc = batch_loss(order[i:i+BATCH], train=True)
            ces.append(ce); accs.append(acc)
        vces, vaccs = [], []
        for i in range(0, len(val_idx), BATCH):
            ce, vm, acc = batch_loss(val_idx[i:i+BATCH], train=False)
            vces.append(ce); vaccs.append(acc)
        print(f"  epoch {ep}: train_ce {np.mean(ces):.3f} acc {np.mean(accs):.3f} | "
              f"val_ce {np.mean(vces):.3f} val_acc {np.mean(vaccs):.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return model


# ---------------------------------------------------------------------------

def main():
    global CONFIG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n_games", nargs="?", type=int, default=8000)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--value-head", choices=["mlp", "attn"], default="mlp")
    ap.add_argument("--teacher", choices=["value", "alphabeta1", "alphabeta2"], default="value")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    CONFIG = Path(args.config)
    n_games = args.n_games
    shards = generate(n_games, teacher=args.teacher, workers=args.workers)

    model, cfg = build_model(value_head=args.value_head)
    model = bc_train(model, cfg, shards)

    run_dir = ROOT / "checkpoints" / "phase2" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "model.zip"))
    model.save(str(run_dir / "self_snapshot.zip"))
    (run_dir / "epsilon.txt").write_text(f"{cfg.train.opponent_cfg.eps_init:.4f}")
    print(f"SEEDED {run_dir}/(model.zip, self_snapshot.zip, epsilon.txt)", flush=True)

    # Eval the BC init
    from catanatron.game import Game
    from catanatron.models.player import Color
    from catan_rl.players.rl_player import RLPlayer
    from catan_rl.eval.opponents import build_opponent
    for opp in ("value", "weighted", "alphabeta1"):
        w = 0
        for s in range(EVAL_GAMES):
            g = Game([RLPlayer(Color.BLUE, run_dir / "model.zip", representation="mixed",
                               enable_maritime_trade=True, deterministic=True),
                      build_opponent(opp, Color.RED)], seed=900 + s)
            g.play()
            w += g.winning_color() == Color.BLUE
        print(f"BC INIT vs {opp}: {w}/{EVAL_GAMES} = {w/EVAL_GAMES:.2f}", flush=True)
    print("BC PRETRAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
