"""Generate (obs, outcome) training data for the OUTCOME-EVALUATOR net.

Why: the PPO value head is poison as a search evaluator (3x confirmed). The fix,
scouted in the Rust-pivot plan, is a dedicated state->outcome regressor trained on
final results of strong games — evaluated exactly at the states search will visit:
the acting seat's decision points.

Setup: VecEnv seats = [policy, value, value, value] at 10 VP — our net vs three
GA-evolved value bots (AB2-class). Seat 0's decision obs are collected (subsampled),
labeled with seat 0's final win/loss when the episode ends, and flushed to .npz
shards. Pure CPU; run for hours/days in the background.

  .venv/bin/python scripts/rust_train/gen_evaluator_data.py \
      --ckpt checkpoints/p10vp/p10vp-tiles-d64-L3.pt --arch tiles --hidden 64 \
      --layers 3 --games 100000 --out data/evaluator
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import catan_py
from nets import build_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", default="tiles")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attn", default="full")
    p.add_argument("--games", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--victory-target", type=int, default=10)
    p.add_argument("--keep-prob", type=float, default=0.25,
                   help="subsample decision states (near-duplicates otherwise)")
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--out", default="data/evaluator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert ck["codec_version"] == catan_py.CODEC_VERSION
    net = build_net(args.arch, catan_py.OBS_DIM, catan_py.NUM_ACTIONS, args.hidden,
                    layers=args.layers, heads=args.heads, attn=args.attn)
    net.load_state_dict(ck["model_state"])
    net.eval()
    print(f"policy: {args.ckpt} @ step {ck['global_step']:,}", flush=True)

    env = catan_py.VecEnv(args.num_envs, victory_target=args.victory_target,
                          visibility="perfect", seed=args.seed,
                          seats=["policy", "value", "value", "value"])
    obs, masks, seats = env.observe()

    pend: dict[int, list] = {i: [] for i in range(args.num_envs)}  # env -> kept obs
    X, y = [], []
    shard = games = wins = 0
    t0 = time.time()

    with torch.no_grad():
        while games < args.games:
            logits, _ = net(torch.as_tensor(obs), torch.as_tensor(masks))
            # sample (not argmax): diversity of visited states matters for a
            # search evaluator; greedy self-cloning collapses the distribution
            acts = torch.distributions.Categorical(logits=logits).sample()
            acts = acts.numpy().astype(np.uint32)
            for i in range(args.num_envs):
                if rng.random() < args.keep_prob:
                    pend[i].append(obs[i].copy())
            obs, masks, seats, rew, dones, terms = env.step(acts)
            for i in range(args.num_envs):
                if dones[i]:
                    won = float(terms[i, 0] > 0)   # seat 0 terminal reward sign
                    for o in pend[i]:
                        X.append(o); y.append(won)
                    pend[i] = []
                    games += 1
                    wins += int(won)
            if len(X) >= args.shard_size:
                f = out / f"shard_{shard:04d}.npz"
                np.savez_compressed(f, X=np.asarray(X, np.float32),
                                    y=np.asarray(y, np.float32))
                print(f"[{time.strftime('%H:%M')}] shard {shard}: {len(X):,} states | "
                      f"games {games:,}/{args.games:,} | policy wr "
                      f"{wins/max(1,games)*100:.1f}% | {games/(time.time()-t0):.1f} g/s",
                      flush=True)
                X, y = [], []
                shard += 1

    if X:
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            X=np.asarray(X, np.float32), y=np.asarray(y, np.float32))
    print(f"DONE: {games:,} games, {shard+1} shards, policy wr {wins/games*100:.1f}%",
          flush=True)


if __name__ == "__main__":
    main()
