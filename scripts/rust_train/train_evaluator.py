"""Train the OUTCOME-EVALUATOR: state -> P(win) regressor.

Data: shards from gen_evaluator_data.py — seat-0 decision obs labeled with the
final win/loss of strong-pool games. Unlike the PPO value head (trained on
bootstrapped returns of a drifting policy — poison in search, 3x confirmed),
this is a plain supervised regressor on REAL outcomes at exactly the states a
search will evaluate.

Model: the tiles-d64 trunk (fast — search calls this per leaf) with a single
sigmoid head. Trains on CPU/MPS in ~1-2h.

  .venv/bin/python scripts/rust_train/train_evaluator.py --data ../../data/evaluator
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nets import TilesHexFormer


class OutcomeEvaluator(nn.Module):
    """tiles-d64 trunk, value-only head (policy heads unused but kept so the
    trunk shape matches; forward returns P(win) for the obs's seat-0)."""

    def __init__(self, hidden=64, layers=3, heads=4):
        super().__init__()
        self.net = TilesHexFormer(1350, 299, hidden, n_layers=layers, n_heads=heads)

    def forward(self, obs):
        mask = torch.ones(obs.shape[0], 299, dtype=torch.bool, device=obs.device)
        _, value = self.net(obs, mask)   # tanh-free head; treat as logit
        return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/evaluator")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-shards", type=int, default=2)
    p.add_argument("--out", default="checkpoints/p10vp/outcome_evaluator_d64.pt")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    shards = sorted(glob.glob(str(Path(args.data) / "shard_*.npz")))
    assert shards, f"no shards under {args.data}"
    val_shards, train_shards = shards[:args.val_shards], shards[args.val_shards:]
    print(f"{len(train_shards)} train shards, {len(val_shards)} val shards", flush=True)

    model = OutcomeEvaluator().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def load(f):
        z = np.load(f)
        return torch.from_numpy(z["X"]), torch.from_numpy(z["y"])

    Xv, yv = zip(*[load(f) for f in val_shards])
    Xv, yv = torch.cat(Xv), torch.cat(yv)

    @torch.no_grad()
    def validate():
        model.eval()
        correct = tot = 0
        losses = []
        for i in range(0, len(Xv), 8192):
            xb, yb = Xv[i:i+8192].to(dev), yv[i:i+8192].to(dev)
            logit = model(xb)
            losses.append(lossf(logit, yb).item())
            correct += ((logit > 0) == (yb > 0.5)).sum().item()
            tot += len(yb)
        model.train()
        return float(np.mean(losses)), correct / tot

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = np.random.default_rng(ep).permutation(len(train_shards))
        for si in order:
            X, y = load(train_shards[si])
            perm = torch.randperm(len(X))
            for i in range(0, len(X), args.batch):
                idx = perm[i:i+args.batch]
                xb, yb = X[idx].to(dev), y[idx].to(dev)
                loss = lossf(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
            if si % 8 == 0:
                vl, va = validate()
                print(f"[{time.strftime('%H:%M')}] ep{ep} shard{si} step {step:,} "
                      f"| val loss {vl:.4f} acc {va*100:.1f}% "
                      f"| {step*args.batch/(time.time()-t0):.0f} samples/s", flush=True)

    vl, va = validate()
    torch.save({"model_state": model.state_dict(), "val_loss": vl, "val_acc": va,
                "arch": "tiles-d64-evaluator", "codec_version": 1, "obs_version": 1},
               args.out)
    print(f"DONE: val loss {vl:.4f}, acc {va*100:.1f}% -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
