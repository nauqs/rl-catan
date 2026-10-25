# Experiments

This is a condensed record of what was tried, why, and what happened, in
roughly chronological order. All work ran on one laptop (Apple M3 Pro, 18 GB),
plus a few Kaggle T4 sessions for the architecture grid.

**How results are reported.** Unless noted:

- Win rates against Catanatron bots are **1v1 games to 10 VP**.
- Rust-engine training metrics are **4-player games** against three copies of a
  bot, where chance is 25%.
- "AB1" and "AB2" are Catanatron's `AlphaBetaPlayer` at depth 1 and 2.
- "Value" is `ValueFunctionPlayer`, a greedy one-ply player using the same
  handcrafted evaluation.

The Catanatron ladder, from weakest to strongest, is random → weighted random
→ value → AB1 → AB2.

## 1. PPO on the Catanatron gym environment

| Setup | vs weighted | vs value | vs AB2 | Games |
|---|---|---|---|---|
| MaskablePPO + BoardCNN, dense reward, 38M steps | 71% | 9% | 7% | 500 |
| + factorized action-type / parameter heads, opponent curriculum | similar | similar | similar | – |
| + spatial pointer head, maritime trade enabled, 50% self-play, 40M steps | 88.5% | 6% | 2.5% | 200 |

The spatial pointer head scores each settlement, road or robber action from
the conv features at that board location, rather than from a pooled vector.
It clearly improved placement: the average pip count of the opening
settlements rose from 5.3 to about 10. It did not move the results against
the strong bots.

## 2. Search and a tile transformer

- **UCB1 root search.** Each move runs UCB1 over the root actions, with short
  self-play rollouts and a value-net bootstrap at the leaves. On the spatial
  PPO policy it raised the vs-value / AB1 / AB2 win rates from 6 / 7 / 2.5%
  to 15.5 / 26 / 16%. Search amplifies a competent policy; it cannot create
  one.
- **HexFormer** (19 tile and 54 vertex tokens, attention masked to board
  adjacency).
  - With the conv-tuned hyper-parameters (lr 3e-4) the optimizer churned:
    KL was 15–25× higher and the run regressed.
  - With lr 1e-4 plus a 1M-step warmup it trained stably, but stayed at
    0–5% against the value bot.
- **Curricula.**
  - An adaptive sampler weighted the "contestable" ladder rungs by w(1−w).
  - The ladder has a gap between the weighted and value bots. To fill it, an
    ε-curriculum plays the value bot with probability ε of a random move,
    and recalibrates ε after each evaluation to hold the agent near 50%.
  - The agent climbed (ε fell from 0.75 to 0.54), but slowly. True vs-value
    win rate stayed near 0.

## 3. Behaviour cloning and fine-tuning

A 6.5-minute behaviour-cloning run beat every multi-day PPO run. It was
trained on 2k value-vs-value games and scored 27% against the value bot and
97% against weighted. The scaled-up version (8k games, 454k decisions,
validation accuracy 0.82) reached **value 47%, AB1 40%, AB2 29%**. The
bottleneck was exploration: PPO does not discover the value bot's tempo by
sampling its own policy. A 5× larger network cloned no better (validation
accuracy 0.810 vs 0.817).

Fine-tuning the clone with RL:

| Variant | Outcome |
|---|---|
| BC → PPO | Improved to AB1 30% / AB2 20% by ~5M steps, then collapsed to 0 by 10M. |
| + critic warmup (train the value head alone, policy frozen) | Stable through the collapse window, but no gain over BC. |
| + lean, win-aligned reward and an attention value head | Improved mid-run (AB2 25%), then slowly collapsed by ~12M. |
| + KL(π ‖ π_BC) anchor, fixed and adaptive coefficient | No collapse, but the agent stays at or slightly below BC. |
| MMD-style moving magnet and imperfect-information-game hyper-parameters (Rudolph et al., 2025) | No improvement over BC. |
| ExIt with a UCB1 teacher | Visit-count targets were too flat to distil (argmax-Q training accuracy 0.32). |

**Diagnosis.** PPO drifts the policy towards whatever the critic over-rates.
The critic is inaccurate off the demonstration manifold, so performance
erodes. The KL anchor protects the floor, but cannot raise the ceiling.

Two long runs reached the same point, one from the BC seed (38M steps) and
one from scratch (40M steps): about 5% against the value bot. Together with
the earlier runs that makes roughly ten PPO runs that all stalled there.

## 4. Policy-pruned alpha-beta search

This variant swaps the leaf evaluation of Catanatron's `AlphaBetaPlayer` and
prunes every node to the policy's top-k moves:

- **The value net as the only leaf evaluation** gives 5%. Minimax seeks out
  off-distribution states that the net over-rates.
- **Policy top-k pruning with the BC value head:**
  - top-3, depth 3: **45% vs AB2** (40 games);
  - at depth 2 (equal to AB2): 33–37%.
- **Retraining the value head** raised held-out explained variance from 0.33
  to 0.64, but it made the search *worse* (AB2 20%).
- **ExIt over this search.** Neither distilling the value from search-state
  outcomes nor distilling the policy from the search's moves helped; playing
  strength fell over rounds in both cases. Accuracy on the teacher's targets
  rose while strength fell.

**Lesson.** Search leaves need strong *local ordinal* fidelity. A handcrafted
evaluation, or true rollouts, provides it; a weakly predictive learned value
does not.

## 5. Rust engine

We vendored the engine from [Eli6th/catan-rl](https://github.com/Eli6th/catan-rl):

- Raw simulation runs at 54k games/s.
- Through the Python `VecEnv` (which releases the GIL) it runs about 554k
  steps/s, compared with ~150 for the Catanatron gym environment.
- We added a value-function seat, using Catanatron's evaluation weights
  ported to the engine's features. It wins 70.8% against three heuristic
  bots; after GA tuning it wins 73.2% (6,000 games, seats rotated).

### Architecture grid

Each model was trained with self-play PPO against heuristic seats, first to
7 VP. The score is the win rate against three heuristic bots, averaged over
the final evaluations.

| Trunk | Width / depth | Params | Steps | Win rate |
|---|---|---|---|---|
| MLP | 256 × 2 | 488k | 500M | 71% |
| Transformer, all-to-all over 146 tokens | d128 / L2–L4 | ~340–490k | 20–67M | 47–60% |
| Transformer, tiles-only attention (Charlesworth-style) | d128 / L2–L4 | | 50–100M | 71–75% |
| Transformer, tiles-only attention | **d64 / L3** | **145k** | 50M | **79%** |
| Transformer, tiles + graph mask | d64–128 / L3–L4 | | 50–100M | 72–78% |
| **Transformer, attention masked to the board graph** | **d128 / L3** | 486k | 76M | **80%** |

- Graph-masked attention matched the MLP's 500M-step result after about 20M
  steps.
- Depth beyond 3 layers did not help.
- Minibatch 2048 added 4–8 points over 1024 in the tiles models.
- The all-to-all transformer learned faster than the MLP early on, but
  plateaued lower.

### Transfer to Catanatron

The adapter (`catan_rl/eval/adapter_maps.py`, `rust_net_player.py`):

- derives the tile, vertex and edge bijections geometrically and checks them
  against the engine's topology;
- rebuilds the engine observation from Catanatron state.

On a fidelity check (one net against three weighted bots) it scores 77%,
against ~95% for the same net inside the engine.

| Network | Raw vs AB2 | + pruned depth-3 search | Games |
|---|---|---|---|
| 4-player net, first to 7 | 27% | 45% | 160 |
| Graph transformer, 10 VP, 17.7M steps | 30% | **47.6%** | 170 |
| + value-bot opponent pool, 60M steps | 16.7% | 42.5% | 120 |
| + self-play league, 100M steps (in-engine: 94% vs heuristic, ~90% vs value) | 25.8% | 37.5% | 120 |

- **In-engine gains do not transfer.** Stronger in-engine play did not move
  the cross-engine number. The policy overfits to its training opponents,
  while search restores most of the lost ranking quality.
- **The rules match.** A side-by-side rules audit found no high-frequency
  divergence between the engines. The one difference is a rare
  bank-depletion edge case.
- **The outcome evaluator failed as a leaf.** It was trained on 9.8M states
  from 300k games, but scored 15.8% as a search leaf, against 41.7% for
  Catanatron's evaluation on the same search (120 games each). That is the
  fourth confirmation of the learned-leaf failure.

## Open directions

- Run the policy network inside the Rust engine, so search can use
  AlphaBot-style full rollouts at engine speed. Upstream reports 85% against
  its heuristic with this recipe.
- Expert iteration with rollout-grounded targets.
- A 2-player fine-tune for the 1v1 ladder.
- Player-to-player trading. The engine has the actions; no policy uses them.
