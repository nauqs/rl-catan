# Catan RL Agent

Training a reinforcement learning agent to play Settlers of Catan from scratch, built on top of [Catanatron](https://docs.catanatron.com). The agent must plug into Catanatron as a standard `Player` subclass so it can be evaluated side-by-side against Catanatron's existing bots.

## North star

Get as close as possible to (and ideally beat) Catanatron's `AlphaBetaPlayer` with `n=2`, the current strongest bot in the Catanatron leaderboard. Earlier phases have more modest, well-defined targets — the AlphaBeta target is the project finish line, not the first checkpoint.

## Scope

**In scope**
- Catanatron's `catanatron/Catanatron-v0` gym environment as the sole simulator
- 1v1 and 4-player game modes (we decide which per experiment)
- Action-masked PPO as the base RL algorithm
- Self-play / league training against past checkpoints and built-in bots
- Multi-head action architecture for composite actions (e.g. play Year of Plenty + pick two resources, build settlement at corner X)
- Evaluation strictly through the Catanatron ladder (Random → WeightedRandom → ValueFunction → AlphaBeta n=1 → AlphaBeta n=2)

**Out of scope (deliberately)**
- **Trading.** Disabled project-wide. Trade-related actions are removed from the action mask so the agent never proposes, accepts, or rejects trades. Charlesworth's settlers-rl project spent ~1 month training a 4-player agent with trading and never converged on sensible trade behavior; we're not repeating that.
- Reimplementing game logic — everything we build wraps or extends Catanatron
- Imitation learning from human games (training is from scratch)
- Distributed / multi-machine training

## Tech stack

- **Python 3.11+** (managed with [`uv`](https://docs.astral.sh/uv/) for environments and dependencies)
- `catanatron[gym]` — environment + benchmarking opponents (installed from the git `main`, since `AlphaBetaPlayer` / `ValueFunctionPlayer` and the bundled `catanatron/Catanatron-v0` gym env aren't shipped in the 3.2.1 PyPI release)
- `sb3-contrib` — `MaskablePPO` as the starting algorithm
- `pytorch` (with MPS or CPU backend on M3 Pro — to be decided per experiment after profiling)
- `wandb` for experiment tracking
- `pytest` for tests

For Phase 3 onwards we may need to drop SB3 and write a CleanRL-style custom PPO loop to support the multi-head policy cleanly. Decision deferred until we get there.

## Suggested repo layout

```
catan-rl/
├── README.md                  # this file
├── pyproject.toml
├── .python-version            # 3.11 (uv-managed)
├── catan_rl/
│   ├── env/                   # env wrappers (no-trade mask, custom observations)
│   ├── features/              # feature extractors (default MLP, tile-attention, etc.)
│   ├── policies/              # SB3 policies + custom multi-head policy
│   ├── players/               # Catanatron Player subclasses wrapping trained models
│   ├── training/              # PPO training entry points, league/self-play
│   ├── eval/                  # benchmark harness, tournament runner
│   └── utils/
├── configs/                   # YAML configs per experiment
├── scripts/                   # train.py, eval.py, play.py, smoke_random.py
├── checkpoints/               # saved models (gitignored)
└── tests/
```

## Evaluation methodology

Every checkpoint we care about gets run through the same harness:

1. **Ladder evaluation.** 500 games against each of: `RandomPlayer`, `WeightedRandomPlayer`, `ValueFunctionPlayer`, `AlphaBetaPlayer(n=1)`, `AlphaBetaPlayer(n=2)`. Report winrate, average victory points, average game length, and 95% confidence interval on winrate.
2. **Run in both 1v1 and 4-player.** 4-player slots: agent + 3 copies of the opponent. Always evaluate both modes for any checkpoint that claims a milestone.
3. **Internal regression.** Play 200 games against the previous best checkpoint of our own agent. Catches silent regressions during long runs.
4. **Log everything to W&B**: per-update loss curves, entropy, value loss, explained variance, eval winrates by opponent and game mode.

The eval harness is built once in Phase 0 and reused unchanged for the rest of the project.

---

## Roadmap

### Phase 0 — Project scaffolding & evaluation harness

**Goal.** Working repo, working environment, working evaluation against all benchmark opponents. No training yet.

**Tasks**
- Set up Python environment with `uv` (`uv venv`, `uv add catanatron[gym] sb3-contrib ...`); all installs and script runs go through `uv`
- Confirm `catanatron/Catanatron-v0` runs end-to-end with a random agent in both 1v1 and 4-player
- Build a **no-trade env wrapper** that strips trade actions from `info["valid_actions"]` and `action_masks()` before the mask is applied. In catanatron 3.3 the gym action array only exposes `MARITIME_TRADE` (player-to-player trades aren't part of the gym action space at all), so that's the only ActionType the wrapper has to filter. Verify by running games and asserting no trade actions are ever taken.
- Implement an **evaluation harness** that takes any callable (random policy, heuristic player, or trained model wrapper) and runs the full ladder eval described above
- Implement a **`Player` subclass wrapper** (`catan_rl.players.RLPlayer`) that loads a trained model from disk and acts in Catanatron games — this is how our agent slots into Catanatron's own tournament infrastructure
- Set up W&B project, base config schema

**Deliverable.** `uv run python -m catan_rl.eval --candidate random --opponent alphabeta2 --games 100 --mode 1v1` returns a winrate. Currently meaningless (no trained model), but the pipeline works. Opponent names: `random`, `weighted`, `value`, `alphabeta1`, `alphabeta2`, or `all`.

---

### Phase 1 — First PPO baseline

**Goal.** Train a vanilla MaskablePPO baseline end-to-end and measure where it lands on the full benchmark ladder. There is no predetermined winrate target — Phase 1's output is *data*, not strength. It tells us where the simple version saturates, which informs everything in Phase 2.

**Setup**
- MaskablePPO with a small custom `BoardCNN` feature extractor (not the default `[64,64]` MLP — that's too small to do anything interesting against the stronger benchmark bots).
- Catanatron's `representation="mixed"` observation: `{board: (channels, 21, 11) hex-grid tensor, numeric: (~48,) global features}`. Built into the env, zero feature-engineering on our side.
- Architecture (~600–800k params): 3 conv layers on the board (16→32→64→128 channels, 3×3 same-padding, ReLU) → global average pool → linear to 256; numeric branch is `48 → 256 → 256`; concat → 512-dim features → SB3 default `[256,256]` actor + critic heads. The board branch is a clean drop-in point for Phase 2 to swap in tile self-attention if the CNN plateaus.
- Flat `Discrete(332)` action space with masking via `MaskablePPO`'s built-in `action_masks()` hook (the `NoTradeWrapper` zeroes out the 60 `MARITIME_TRADE` indices).
- 1v1 vs `WeightedRandomPlayer` as the training opponent. Plain `RandomPlayer` is too weak (rarely punishes positional mistakes, so the agent plateaus low); `ValueFunctionPlayer` / `AlphaBeta` are too strong (sparse ±1 collapses to all-losses, gradient dies). Weighted picks randomly most of the time but actually buys cities/settlements/dev-cards when it can — enough opposition to force real positional play, zero extra infra. Self-play and league training stay in Phase 2.
- Charlesworth-style reward shaping, his exact values (from `settlers_of_catan_RL/env/wrapper.py`): **+500** terminal win (loss = 0, positive-only), **+5** per VP gained, **+5** for playing any development card, **+1** for moving the robber, **−0.3** per forced discard, **+2.5** additional bonus for upgrading to a city (on top of the +5 from the VP delta, so cities = +7.5 total — deliberately biases toward city upgrades). Replaces Catanatron's default ±1 entirely. Implemented as `ShapedRewardWrapper`. Dense-reward annealing (Charlesworth's `reward_annealing_factor`) is also wired — linear taper of dense rewards from 1.0 → 0.0 between two configurable env-step bounds; terminal +500 is unaffected. Defaults are `-1` (disabled, matching his defaults); flipping `anneal_start_step` / `anneal_end_step` to positive values enables it.
- Reasonable PPO starting hyperparameters: SB3 defaults with `gamma=0.999`, `gae_lambda=0.95`, `n_steps=2048`, `batch_size=512`, `ent_coef=0.01`

**Tasks**
- Wire up MaskablePPO with `SubprocVecEnv` (`NoTradeWrapper` exposes `action_masks()` on the env so SubprocVecEnv can reach it via `env_method`)
- CPU vs MPS profiled (`scripts/profile_device.py`): with the BoardCNN architecture on M3 Pro, MPS wins decisively — ~397 steps/sec vs ~130 on CPU (3× speedup). `configs/phase1.yaml` defaults to `device: mps`. The MPS gain comes from the conv ops being big enough that per-call shipping overhead amortizes; for the original 50k-param MLP it would have been the opposite.
- Train (10M timesteps minimum, ~5–15h on M3 Pro CPU), checkpoint every ~100k steps, run a lightweight ladder eval (Random + Weighted + Value, 20 games each) every ~50k steps during training, then a full 500-game × 5-opponent × 1v1 ladder at the end
- Crash-safe resume: each run dir holds `model.zip` + `wandb_run_id.txt` so `scripts/train.py --resume` reloads the policy and reattaches the same W&B run (`resume="must"`) — the plot continues unbroken
- Document where the baseline plateaus and what the agent's failure modes look like — settlement placement quality, robber behaviour, dev card play, dice-roll handling, etc.

**Deliverables**
- One or more trained checkpoints with logged ladder-eval results across both game modes
- A clear, honest picture of how far vanilla MaskablePPO gets without any of the more complex ideas. This is the reference point everything in Phase 2 is compared against.
- Notes on observed failure modes to inform which Phase 2 directions are worth pursuing

---

### Phase 2 — Improvements (open-ended)

Phase 2 is deliberately not a fixed sequence. The actual direction is driven by what we see in Phase 1: which failure modes dominate, where the loss curves plateau, whether the agent looks credit-assignment limited or representation limited, etc. Below is the menu of ideas worth considering — roughly ordered by typical cost-to-leverage ratio, but the real order and combination is something we decide with Phase 1 results in hand. We can also revisit Phase 2 multiple times, picking up additional ideas as earlier ones plateau.

**Possible directions, in no fixed order**

- **Custom observation features.** Catanatron's default feature vector may or may not give the policy enough structure to learn from. Candidates to encode explicitly: one-hot tile dice values (no implicit ordering between 6 and 8), tile resource type, robber position, per-corner ownership encoded relative to the current player, current-player resource *production* per dice value, opponent state with hidden info masked (min/max bounds), bank state. Built as a custom `BaseFeaturesExtractor` in PyTorch.

- **Reward shaping.** A sparse +1/-1 signal over ~150 decisions per game is weak. Add small dense rewards for building, gaining VPs, playing knights, moving the robber, with magnitudes tuned so they can't outweigh the terminal win reward. Often a large lift when the baseline struggles with long-horizon credit assignment.

- **Multi-head action architecture.** Replace the flat ~290-way softmax with an action-type head plus conditional parameter heads (corner / edge / tile / dev-card / resource / steal-target). Lets a single decision express composite actions like "play Year of Plenty + pick wheat + pick wood" cleanly, and lets credit assignment flow to the type decision independently of the parameter choices. Highest engineering cost in this menu — likely requires moving from SB3 to a custom CleanRL-style PPO loop since SB3's policy API doesn't compose cleanly with conditional sampling.

- **Self-play / league training.** Train against a pool of opponents (`RandomPlayer`, `WeightedRandomPlayer`, `ValueFunctionPlayer`, `AlphaBetaPlayer(n=1)`, plus recent self-checkpoints with recency-biased sampling) instead of a single fixed opponent. Reduces overfitting and pushes the agent to be robust across play styles. Loss-weighted sampling can focus training on opponents the agent is currently bad against.

- **4-player training.** Up to Phase 1 4-player is eval-only. Bringing 4-player into the training rollouts is its own lever — 3× more decisions per game and a non-stationary mix of three opponents, but possibly the only way to be strong in 4-player given that 1v1 strategies don't always transfer.

- **Tile attention encoder.** Small self-attention over the 19 hexes inside the feature extractor (Charlesworth's approach). Whether it actually helps at our compute budget is an empirical question — probably worth trying only after a non-attention custom extractor has been benchmarked.

- **Inference-time root-level search.** Single-step UCB1 expansion at the root with policy rollouts. Pure inference-time — no retraining needed. Charlesworth saw a substantial winrate boost from this (25% → 47% against three copies of itself). Requires implementing game state snapshot/restore and randomization of opponents' hidden hands consistent with public information.

- **Other ideas worth keeping in mind.** Value function normalization, recomputing GAE returns between PPO epochs, entropy annealing, scaling up network size if MPS turns out to be worthwhile, exploring different `gamma` values for long-horizon credit assignment.

**How to pick what to do**
- Start with whatever Phase 1's failure modes most clearly point at. Plateau early in training → try reward shaping or feature engineering first. Looks like the agent can't express the action it wants → multi-head is a strong candidate. Wins against random but loses badly to stronger opponents → league training.
- Prefer cheap interventions before expensive ones. Reward shaping and feature engineering land before custom PPO loops.
- Re-run the full ladder eval after every meaningful change so we always know whether the latest move actually improved things or made them worse.
- Multi-head and league training are both plausible "biggest lever" choices, but we should earn that conclusion from data rather than assume it.

---

## Decisions still to make (parking lot)

- **Discount factor** for the long-horizon Catan reward structure. Default 0.999 in Phase 1, worth revisiting in Phase 2.
- **Reward shaping coefficients** — if pursued in Phase 2, magnitudes are set heuristically and we likely won't have the compute budget for proper ablations.
- **MPS vs CPU** for the policy network — profile in Phase 1 and revisit if network size grows significantly.
- **Custom PPO loop vs SB3 subclass** — only relevant if Phase 2 pursues multi-head. First attempt at SB3 subclass; if it gets ugly, move to CleanRL-style.
- **4-player branching factor** — adds wall-clock cost. Decide per experiment whether to include 4-player in training rollouts or only in eval.
- **Whether to sample `AlphaBetaPlayer(n=2)` as a training opponent** — it's our eval target. Training against it might be either highly effective or destructively imbalanced.

## References

- **Catanatron** — [docs](https://docs.catanatron.com), [GitHub](https://github.com/bcollazo/catanatron). Our base simulator and benchmark suite.
- **Charlesworth, "Learning to Play Settlers of Catan with Deep RL"** — [project page](https://settlers-rl.github.io), [code](https://github.com/henrycharlesworth/settlers_of_catan_RL). Main inspiration for the multi-head action architecture, tile encoder, reward shaping, and league training. Trained on 32-core + 128GB + GTX 3090 for ~1 month; we are working with ~1/100th that compute and adjusting scope accordingly.
- **Schulman et al. 2017, "Proximal Policy Optimization Algorithms"** — base RL algorithm. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **Andrychowicz et al. 2020, "What Matters in On-Policy Reinforcement Learning?"** — useful for PPO hyperparameter choices. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
- **Liu et al. 2018, "WU-UCT"** — parallel MCTS variant used for the Phase 5 forward search. [arXiv:1810.11755](https://arxiv.org/abs/1810.11755)
- **sb3-contrib MaskablePPO** — [docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). Phase 1–2 training loop.
