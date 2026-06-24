"""Train ONLY the value head, longer, supervised (trunk + policy frozen).

Hypothesis: BC under-cooks the value head. The policy cross-entropy is an
easy target that converges fast; the value MSE target (high-variance discounted
win/loss, weighted 0.3, only 6 epochs in bc_pretrain) may need much more. So:
freeze trunk + policy, train value_net for many epochs on value-vs-value outcomes,
and watch explained-variance (EV) climb (or plateau, = already converged).

Then re-eval NeuralAlphaBeta with the improved value net.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch as th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default="checkpoints/bc_seed/model.zip")
    ap.add_argument("--out", default="checkpoints/bc_seed/model_vtrain.zip")
    ap.add_argument("--n-games", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.999)
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from catan_rl.players.rl_player import _force_cpu_deserialize
    from bc_pretrain import generate  # noqa: E402  (scripts/ on path)

    shards = generate(args.n_games, teacher="value", workers=6)
    data = [np.load(p) for p in shards]
    boards = np.concatenate([d["boards"] for d in data])
    numerics = np.concatenate([d["numerics"] for d in data])
    dts = np.concatenate([d["dts"] for d in data])
    wons = np.concatenate([d["wons"] for d in data])
    del data
    n = len(dts)
    vtarget = (args.gamma ** dts) * np.where(wons == 1, 500.0, -500.0)
    print(f"value dataset: {n} positions ({boards.nbytes/1e9:.1f} GB boards)", flush=True)

    with _force_cpu_deserialize():
        model = MaskablePPO.load(args.seed_model, device="mps")
    pol = model.policy
    pol.set_training_mode(True)
    n_train = 0
    for name, p in pol.named_parameters():
        train = name.startswith("value_net")
        p.requires_grad_(train)
        n_train += int(train)
    vparams = [p for nm, p in pol.named_parameters() if nm.startswith("value_net")]
    opt = th.optim.Adam(vparams, lr=args.lr)
    dev = model.device
    print(f"trainable value-head tensors: {n_train}", flush=True)

    rng = np.random.default_rng(0)
    holdout = max(1, int(0.03 * n))
    perm0 = rng.permutation(n)
    val_idx, train_idx = perm0[:holdout], perm0[holdout:]

    def run_batch(idx, train: bool):
        obs = {"board": th.tensor(boards[idx], dtype=th.float32, device=dev),
               "numeric": th.tensor(numerics[idx], dtype=th.float32, device=dev)}
        vt = th.tensor(vtarget[idx], dtype=th.float32, device=dev)
        feats = pol.extract_features(obs)
        fm, g = pol._split(feats)
        v = pol._value(fm, g).squeeze(-1)
        loss = th.nn.functional.mse_loss(v / 500.0, vt / 500.0)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss), v.detach().cpu().numpy(), vt.detach().cpu().numpy()

    t0 = time.time()
    # baseline EV before any extra training
    for label, do_train in [("epoch -1 (baseline)", False)]:
        vs, vts, ls = [], [], []
        for i in range(0, len(val_idx), args.batch):
            l, vp, vtt = run_batch(val_idx[i:i + args.batch], False)
            ls.append(l); vs.append(vp); vts.append(vtt)
        vs = np.concatenate(vs); vts = np.concatenate(vts)
        ev = 1 - np.var(vts - vs) / (np.var(vts) + 1e-8)
        print(f"{label}: val_mse {np.mean(ls):.4f} EV {ev:.3f}", flush=True)

    for ep in range(args.epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), args.batch):
            run_batch(order[i:i + args.batch], True)
        vs, vts, ls = [], [], []
        for i in range(0, len(val_idx), args.batch):
            l, vp, vtt = run_batch(val_idx[i:i + args.batch], False)
            ls.append(l); vs.append(vp); vts.append(vtt)
        vs = np.concatenate(vs); vts = np.concatenate(vts)
        ev = 1 - np.var(vts - vs) / (np.var(vts) + 1e-8)
        print(f"epoch {ep}: val_mse {np.mean(ls):.4f} EV {ev:.3f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        if (ep + 1) % 10 == 0:
            model.save(args.out)
    model.save(args.out)
    print(f"SAVED {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
