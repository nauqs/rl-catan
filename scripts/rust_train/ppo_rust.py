"""Masked PPO self-play on the Catan VecEnv (Phase 10 trainer).

One policy plays all seats (seat-relative obs). Trajectories are stitched
per (env, seat): a transition completes when that seat next acts — at that
moment we know both its accrued reward and its bootstrap value — or at
episode end via terminal_rewards.

ATTRIBUTION: the PPO/self-play loop and PolicyValueNet below are adapted from
Eli6th/catan-rl (MIT). The Rust engine (catan_py) is vendored under rust/ with
UPSTREAM_LICENSE + VENDORED_FROM.md. This file adds: Weights & Biases logging,
a step-based budget, and our project-standard run conventions so the Rust-engine
runs land in the same wandb project as Phases 1-9. See LOGBOOK.md "Phase 10".

Stage-1 smoke run (first-to-7, perfect info, shaped reward):
    python scripts/rust_train/ppo_rust.py --name smoke1 --minutes 10 \
        --victory-target 7 --vp-delta 0.05

Long run with wandb (Phase 10):
    python scripts/rust_train/ppo_rust.py --name phase10-rust \
        --steps 200_000_000 --num-envs 256 \
        --train-seats policy,heuristic,policy,heuristic \
        --victory-target 7 --vp-delta 0.05 --wandb
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import catan_py

from nets import build_net  # scripts/rust_train is on sys.path[0] when run directly

try:
    import wandb
except ImportError:  # wandb optional; --wandb will error clearly if missing
    wandb = None

GAMMA = 1.0  # episodic, bounded: do NOT discount 600-step credit away
GAE_LAMBDA = 0.95
CLIP = 0.2
DEFAULT_ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=0,
                   help="env-step budget; if >0 it overrides --minutes "
                        "(run until global_step >= steps)")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="hard wall-clock cap in hours; stop at min(steps, cap) "
                        "with a final checkpoint (fits Kaggle's 12h kernel limit)")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-players", type=int, default=4, choices=[2, 3, 4],
                   help="players per game (2 = heads-up fine-tune for the 1v1 "
                        "ladder; --train-seats must list exactly this many)")
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--victory-target", type=int, default=7)
    p.add_argument("--vp-delta", type=float, default=0.05)
    p.add_argument("--visibility", default="perfect")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--arch", default="mlp", choices=["mlp", "hexformer", "tiles"],
                   help="mlp = baseline 2-layer net; hexformer = Tile-Transformer "
                        "trunk + spatial pointer head (use --hidden ~128); "
                        "tiles = Charlesworth-faithful tiles-only attention + scatter")
    p.add_argument("--attn", default="full", choices=["full", "graph"],
                   help="hexformer attention: full (all-to-all 146 tokens) or "
                        "graph (masked to real board adjacency + global hub)")
    p.add_argument("--layers", type=int, default=3, help="hexformer encoder layers")
    p.add_argument("--heads", type=int, default=4, help="hexformer attention heads")
    p.add_argument("--warmup-updates", type=int, default=0,
                   help="linear LR warmup 0->lr over the first N updates "
                        "(transformers need this; standing conclusion #7)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4,
                   help="torch CPU threads (raise for the heavier hexformer trunk)")
    p.add_argument("--eval-every", type=int, default=8)
    p.add_argument("--eval-games", type=int, default=96,
                   help="episodes per eval opponent (value-bot games at 10VP are "
                        "slow — lower this when value-3 is in the eval set)")
    p.add_argument("--metrics", default=None, help="metrics JSONL path override")
    p.add_argument("--resume", default=None, help="checkpoint to warm-start from")
    p.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    p.add_argument("--vp-delta-final", type=float, default=None,
                   help="anneal shaped reward toward this over 3 run phases")
    p.add_argument("--train-seats", default=None,
                   help="comma list, e.g. policy,heuristic,policy,heuristic — "
                        "mixed-opponent training (default: all policy self-play)")
    # --- Archive league (added after the fixed value-seat pool failed to transfer:
    # a fixed sparring pool gets EXPLOITED, not learned from — Jul 8) ----------
    p.add_argument("--league", action="store_true",
                   help="drive --league-seats with frozen past-self snapshots "
                        "instead of the live net (population play)")
    p.add_argument("--league-size", type=int, default=8,
                   help="ring size of archived past selves")
    p.add_argument("--league-snapshot-every", type=int, default=150,
                   help="updates between archive snapshots (~5M steps at 256x128)")
    p.add_argument("--league-seats", default="2",
                   help="comma seat indices played by the archive (these seats "
                        "must be 'policy' in --train-seats; they are NOT trained)")
    p.add_argument("--wandb", action="store_true",
                   help="log train/eval metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="catan-rl")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args()


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.policy(h)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)


class Metrics:
    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
        self.emit({"t": "run", **header})

    def emit(self, event: dict):
        event["unix_ms"] = int(time.time() * 1000)
        self.f.write(json.dumps(event) + "\n")


def save_checkpoint(run_dir: Path, net, optimizer, step: int, config: dict):
    ck_dir = run_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": step,
        "config": config,
        "codec_version": catan_py.CODEC_VERSION,
        "num_actions": catan_py.NUM_ACTIONS,
        "obs_version": catan_py.OBS_VERSION,
        "obs_dim": catan_py.OBS_DIM,
        "engine_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    path = ck_dir / f"step_{step:010d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic
    latest = ck_dir / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)


@torch.no_grad()
def evaluate_vs(net, device, opponent: str, episodes: int, seed: int, args) -> dict:
    """Greedy policy in seat 0 vs num_players-1 engine bots ('random' or
    'heuristic'). Bot seats play inside the env at engine speed, so every
    surfaced decision belongs to the policy. Baseline = 1/num_players."""
    n = getattr(args, "num_players", 4)
    env = catan_py.VecEnv(
        48,
        num_players=n,
        victory_target=args.victory_target,
        visibility=args.visibility,
        seed=seed,
        seats=["policy"] + [opponent] * (n - 1),
    )
    obs, masks, seats = env.observe()
    wins = done_eps = vp_sum = caps = 0
    while done_eps < episodes:
        o = torch.as_tensor(obs, device=device)
        m = torch.as_tensor(masks, device=device)
        logits, _ = net(o, m)
        actions = logits.argmax(dim=1).cpu().numpy().astype(np.uint32)
        obs, masks, seats, rewards, dones, terminals = env.step(actions)
        for turns, winner, vps, cap in env.take_episode_stats():
            done_eps += 1
            wins += int(winner == 0)
            vp_sum += vps[0]
            caps += int(cap)
    return {
        "win_rate": wins / done_eps,
        "games": done_eps,
        "avg_vp": vp_sum / max(1, done_eps),
        "cap_rate": caps / max(1, done_eps),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    run_dir = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    use_wandb = args.wandb
    if use_wandb:
        if wandb is None:
            raise SystemExit("--wandb passed but the wandb package is not installed")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            config={**config, "engine": "rust", "phase": 10,
                    "obs_dim": catan_py.OBS_DIM, "num_actions": catan_py.NUM_ACTIONS,
                    "codec_version": catan_py.CODEC_VERSION,
                    "obs_version": catan_py.OBS_VERSION},
            dir=str(run_dir),
        )
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.jsonl"
    seat_names = (args.train_seats.split(",") if args.train_seats
                  else ["Policy"] * args.num_players)
    metrics = Metrics(metrics_path, {"source": "ppo", "games": 0,
                                     "players": [s.capitalize() for s in seat_names],
                                     "seed": args.seed, "run_dir": str(run_dir)})

    train_seats = args.train_seats.split(",") if args.train_seats else None
    if train_seats:
        assert len(train_seats) == args.num_players, \
            f"--train-seats lists {len(train_seats)} seats but --num-players={args.num_players}"
    env = catan_py.VecEnv(
        args.num_envs,
        num_players=args.num_players,
        victory_target=args.victory_target,
        visibility=args.visibility,
        vp_delta=args.vp_delta,
        seed=args.seed,
        seats=train_seats,
    )
    net = build_net(args.arch, env.obs_dim, env.num_actions, args.hidden,
                    layers=args.layers, heads=args.heads, attn=args.attn).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    resume_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        assert ck["codec_version"] == catan_py.CODEC_VERSION, "codec version mismatch"
        assert ck["obs_version"] == catan_py.OBS_VERSION, "obs version mismatch"
        net.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        resume_step = int(ck["global_step"])
        print(f"resumed from {args.resume} at step {resume_step:,} "
              f"(--steps is the TOTAL target; warmup counts from resume)")
    print(f"net params: {sum(p.numel() for p in net.parameters()):,} | device {device}")
    print(f"run dir: {run_dir}\nmetrics: {metrics_path}")

    # Archive league: a second net instance replays frozen snapshots on the
    # league seats. Learner seats keep training; league seats are opponents
    # only (their transitions are never recorded).
    league_seatset: set[int] = set()
    league_net = None
    league_archive: list[dict] = []
    if args.league:
        league_seatset = {int(x) for x in args.league_seats.split(",")}
        league_net = build_net(args.arch, env.obs_dim, env.num_actions, args.hidden,
                               layers=args.layers, heads=args.heads,
                               attn=args.attn).to(device)
        league_net.load_state_dict(net.state_dict())
        league_net.eval()
        league_archive.append(
            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        print(f"league: seats {sorted(league_seatset)} vs archive of "
              f"{args.league_size} (snapshot every {args.league_snapshot_every} updates)")

    obs, masks, seats = env.observe()
    # Per-(env, seat) chains: pending transition awaiting reward/next value.
    pending = {}
    chains = defaultdict(list)  # (env, seat) -> completed transitions
    global_step = resume_step
    update = 0
    start_time = time.time()
    # Budget: stop at whichever comes first — the step target or a wall-clock cap.
    # --max-hours is a hard ceiling that coexists with --steps (fits runs inside
    # Kaggle's 12h kernel limit: light cells hit --steps first, heavy 146-token
    # cells stop at the cap with a final checkpoint). --minutes is the legacy cap
    # used only when no step target is set.
    if args.max_hours > 0:
        deadline = start_time + args.max_hours * 3600
    elif args.steps > 0:
        deadline = float("inf")
    else:
        deadline = start_time + args.minutes * 60
    # Shaped-reward annealing: 3 equal phases interpolating vp_delta toward
    # --vp-delta-final (the honest objective takes over by the last phase).
    anneal = None
    if args.vp_delta_final is not None:
        anneal = [args.vp_delta,
                  (args.vp_delta + args.vp_delta_final) / 2,
                  args.vp_delta_final]
    current_phase = 0

    while time.time() < deadline and (args.steps == 0 or global_step < args.steps):
        update += 1
        # Linear LR warmup (transformers churn under a cold-start hot LR).
        if args.warmup_updates > 0:
            warm = min(1.0, update / args.warmup_updates)
            for grp in optimizer.param_groups:
                grp["lr"] = args.lr * warm
        cur_lr = optimizer.param_groups[0]["lr"]
        if anneal:
            phase = min(int((time.time() - start_time) / (args.minutes * 60 / 3)), 2)
            if phase != current_phase:
                current_phase = phase
                print(f"=== anneal phase {phase}: vp_delta -> {anneal[phase]} ===")
                metrics.emit({"t": "run", "source": "ppo",
                              "players": [s.capitalize() for s in seat_names],
                              "seed": args.seed, "note": f"anneal vp_delta={anneal[phase]}"})
                env = catan_py.VecEnv(
                    args.num_envs, num_players=args.num_players,
                    victory_target=args.victory_target,
                    visibility=args.visibility, vp_delta=anneal[phase],
                    seed=args.seed + phase, seats=train_seats,
                )
                pending.clear()
                chains.clear()
                obs, masks, seats = env.observe()
        t0 = time.time()
        # ----------------------------------------------------- rollout
        with torch.no_grad():
            for _ in range(args.rollout):
                o = torch.as_tensor(obs, device=device)
                m = torch.as_tensor(masks, device=device)
                logits, values = net(o, m)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
                acts_np = acts.cpu().numpy().astype(np.uint32)
                values_np = values.cpu().numpy()
                logps_np = logps.cpu().numpy()

                # League seats act from a frozen snapshot, not the live net.
                if league_net is not None:
                    lg = [i for i in range(args.num_envs)
                          if int(seats[i]) in league_seatset]
                    if lg:
                        llogits, _ = league_net(o[lg], m[lg])
                        lacts = torch.distributions.Categorical(logits=llogits).sample()
                        acts_np[lg] = lacts.cpu().numpy().astype(np.uint32)

                # Open a transition per row; complete the seat's previous one
                # (its reward was assigned when this decision point arrived).
                for i in range(args.num_envs):
                    if int(seats[i]) in league_seatset:
                        continue          # opponents: never recorded, never trained
                    key = (i, int(seats[i]))
                    rec = {
                        "obs": obs[i].copy(), "mask": masks[i].copy(),
                        "action": int(acts_np[i]), "logp": float(logps_np[i]),
                        "value": float(values_np[i]),
                        "reward": 0.0, "done": False, "next_value": 0.0,
                    }
                    prev = pending.get(key)
                    if prev is not None and prev.get("reward_set"):
                        prev["next_value"] = rec["value"]
                        chains[key].append(prev)
                    pending[key] = rec

                obs, masks, seats, rewards, dones, terminals = env.step(acts_np)
                global_step += args.num_envs

                for i in range(args.num_envs):
                    if dones[i]:
                        for s in range(4):
                            prev = pending.pop((i, s), None)
                            if prev is not None:
                                prev["reward"] = float(terminals[i, s])
                                prev["reward_set"] = True
                                prev["done"] = True
                                prev["next_value"] = 0.0
                                chains[(i, s)].append(prev)
                    else:
                        key = (i, int(seats[i]))
                        prev = pending.get(key)
                        if prev is not None:
                            prev["reward"] = float(rewards[i])
                            prev["reward_set"] = True

                for turns, winner, vps, cap in env.take_episode_stats():
                    metrics.emit({"t": "game", "i": 0, "winner": int(winner),
                                  "turns": int(turns), "steps": 0,
                                  "vp": [int(v) for v in vps], "cap": bool(cap)})

        # ------------------------------------------- GAE per chain + flush
        batch = {k: [] for k in ("obs", "mask", "action", "logp", "value", "adv", "ret")}
        for key, chain in chains.items():
            if not chain:
                continue
            boot = pending.get(key)
            next_adv = 0.0
            for idx in range(len(chain) - 1, -1, -1):
                tr = chain[idx]
                nv = tr["next_value"]
                if idx == len(chain) - 1 and not tr["done"] and boot is not None:
                    nv = boot["value"]  # bootstrap from the seat's open decision
                nonterminal = 0.0 if tr["done"] else 1.0
                delta = tr["reward"] + GAMMA * nv * nonterminal - tr["value"]
                adv = delta + GAMMA * GAE_LAMBDA * nonterminal * next_adv
                next_adv = adv
                batch["obs"].append(tr["obs"]); batch["mask"].append(tr["mask"])
                batch["action"].append(tr["action"]); batch["logp"].append(tr["logp"])
                batch["value"].append(tr["value"]); batch["adv"].append(adv)
                batch["ret"].append(adv + tr["value"])
            chain.clear()

        n = len(batch["action"])
        if n == 0:
            continue
        b_obs = torch.as_tensor(np.array(batch["obs"]), device=device)
        b_mask = torch.as_tensor(np.array(batch["mask"]), device=device)
        b_act = torch.as_tensor(batch["action"], device=device)
        b_logp = torch.as_tensor(batch["logp"], device=device)
        b_val = torch.as_tensor(batch["value"], device=device)
        b_adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # --------------------------------------------------- PPO update
        clip_fracs, entropies = [], []
        idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch):
                mb = idx[start:start + args.minibatch]
                logits, values = net(b_obs[mb], b_mask[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * ratio.clamp(1 - CLIP, 1 + CLIP)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * (values - b_ret[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + VALUE_COEF * value_loss - args.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                clip_fracs.append(((ratio - 1).abs() > CLIP).float().mean().item())
                entropies.append(entropy.item())

        # League maintenance: archive the learner periodically; resample which
        # frozen self the league seats replay each update (population variety).
        if league_net is not None:
            if update % args.league_snapshot_every == 0:
                league_archive.append(
                    {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                if len(league_archive) > args.league_size:
                    league_archive.pop(0)
            league_net.load_state_dict(
                league_archive[np.random.randint(len(league_archive))])

        var_ret = float(b_ret.var())
        explained = 1.0 - float((b_ret - b_val).var()) / (var_ret + 1e-8)
        sps = args.rollout * args.num_envs / (time.time() - t0)
        train_metrics = {
            "t": "train", "step": global_step, "entropy": float(np.mean(entropies)),
            "explained_variance": explained, "clip_frac": float(np.mean(clip_fracs)),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()), "lr": cur_lr, "sps": sps,
        }
        metrics.emit(train_metrics)
        if use_wandb:
            wandb.log({
                "train/entropy": train_metrics["entropy"],
                "train/explained_variance": explained,
                "train/clip_frac": train_metrics["clip_frac"],
                "train/policy_loss": train_metrics["policy_loss"],
                "train/value_loss": train_metrics["value_loss"],
                "train/sps": sps, "train/lr": cur_lr,
                "train/batch": n, "train/update": update,
            }, step=global_step)
        print(f"update {update} | step {global_step:,} | {sps:,.0f} sps | "
              f"ent {np.mean(entropies):.2f} | ev {explained:.2f} | "
              f"clip {np.mean(clip_fracs):.2f} | batch {n:,}")

        if update % args.eval_every == 0:
            for opponent, label in (("random", "random-3"), ("heuristic", "heuristic-v1-3"), ("value", "value-3")):
                result = evaluate_vs(net, device, opponent, episodes=args.eval_games,
                                     seed=10_000 + update, args=args)
                metrics.emit({"t": "eval", "step": global_step, "vs": label, **result})
                if use_wandb:
                    wandb.log({
                        f"eval/{opponent}_win_rate": result["win_rate"],
                        f"eval/{opponent}_avg_vp": result["avg_vp"],
                        f"eval/{opponent}_cap_rate": result["cap_rate"],
                    }, step=global_step)
                print(f"  eval vs 3 {opponent}: {result['win_rate']*100:.1f}% "
                      f"({result['games']} games, baseline 25%)")
            save_checkpoint(run_dir, net, optimizer, global_step, config)

    save_checkpoint(run_dir, net, optimizer, global_step, config)
    for opponent, label in (("random", "random-3"), ("heuristic", "heuristic-v1-3"), ("value", "value-3")):
        result = evaluate_vs(net, device, opponent, episodes=192, seed=999, args=args)
        metrics.emit({"t": "eval", "step": global_step, "vs": label, **result})
        if use_wandb:
            wandb.log({
                f"eval/{opponent}_win_rate": result["win_rate"],
                f"eval/{opponent}_avg_vp": result["avg_vp"],
                f"eval/{opponent}_cap_rate": result["cap_rate"],
            }, step=global_step)
        print(f"final eval vs 3 {opponent}: {result['win_rate']*100:.1f}% | "
              f"avg VP {result['avg_vp']:.1f} | done at step {global_step:,}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
