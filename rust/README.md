# Rust engine (`catan-core`) and simulator (`catan-sim`)

The source-of-truth Catan engine: full base-game rules including
player-to-player trading, validated by a rulebook scenario suite,
property-based tests with oracles, and self-golden replays. Built for RL
workloads — every action is mask-validated, every rejection is side-effect
free, and bulk simulation runs tens of thousands of games per second.

## Layout

- `catan-core/` — pure rules engine (state, board topology, building,
  trading, dev cards, robber, turn state machine) plus `RandomPlayer` /
  `HeuristicPlayer` behind a `Player` trait. No I/O.
- `catan-sim/` — parallel simulation CLI (rayon), plus the self-golden
  fixture recorder.
- `catan-env/` — RL environment layer (in progress). Done: the action codec
  (`codec.rs`) — a fixed 299-id discrete action space with seat-relative
  encoding and an exact legality mask. The id layout is documented in the
  module header and pinned by `tests/codec.rs`: layout spot checks,
  encode/decode roundtrip + uniqueness on live games, and a mask⇔engine
  equivalence fuzz (every id at sampled states must execute iff masked
  legal, 3- and 4-player). Also the observation encoder (`obs.rs`): a
  1,350-float seat-relative view (documented layout, values normalized to
  ~[0,1], dice as production probabilities) with `Perfect` and `Realistic`
  visibility modes differing in exactly the opponent-private block; pinned
  by `tests/obs.rs` (layout spot checks, seat-relativity, visibility diff,
  determinism + bounds on live games). Both layouts are FROZEN once a
  network trains against them (`CODEC_VERSION` / `OBS_VERSION` stored in
  checkpoints).

  The environment itself (`env.rs`): `CatanEnv` wraps one game with
  PettingZoo-AEC semantics — `step(action_id)` returns the next seat to
  act, its accrued reward, and on termination a per-seat
  `terminal_rewards` vector. Forced moves (one legal action) are resolved
  internally (`auto_resolve_forced`; tested property: every policy
  decision has >= 2 choices), dice stay internal chance nodes, rewards are
  terminal win/loss plus an optional shaped VP-delta (tested to account
  exactly: total delivered = coefficient x final VP). Built-in CTRP
  episode recording for eval/video. `VecCatanEnv` steps N envs in one
  rayon-parallel call with auto-reset and deterministic per-episode seed
  streams — the unit the trainer talks to. Measured: ~346 ns per full RL
  step (mask + step + obs encode) single-threaded; ~300 us per 1,024-env
  batch (~3.4M policy-steps/s, memory-bandwidth-bound on observation
  writes — far above what NN inference will consume). Hardened by
  `tests/env_zero_alloc.rs` (0 allocations per decision in steady state;
  batch allocations bounded, never per-env) and `tests/env_soak.rs`
  (250-episode soak with reward invariants, per-seed batch determinism,
  env outputs cross-checked against codec/encoder, loud failure modes).
- `catan-py/` — PyO3 bindings (excluded from the cargo workspace; built
  with maturin against a Python toolchain):

  ```bash
  pip install maturin
  cd rust/catan-py && maturin develop --release   # (CONDA_PREFIX or venv)
  python training/smoke_env.py                    # full-contract smoke test
  ```

  `catan_py.VecEnv(num_envs, ...)` steps the whole batch in ONE Python
  call with the GIL released (`observe()` / `step(actions)` over NumPy
  arrays; ~370k policy-steps/s through Python at batch 256).
  `enable_recording()` + `take_replays()` harvest CTRP bytes for the
  video pipeline; `codec_version`/`obs_version` getters feed the
  checkpoint gate. `catan_py.Env` is the single-game handle for eval
  loops. Next: PPO self-play script + eval/Elo harness.

The engine's win threshold is per-game (`GameState::victory_target`,
default 10, `CatanGame::new_with_target`): curriculum training starts on
shorter first-to-7 games. CTRP replays persist and re-enforce the target.

## Rules coverage

Complete base game (3-4 players; 2 supported): setup snake draft, dice +
resource distribution with bank-shortage rules, building with distance and
connectivity rules, dev cards (one per turn, none on the turn bought, VP
cards auto-scored), robber with sequential player-chosen discards, bank
trading at 4:1 / 3:1 / 2:1 via per-game shuffled ports, longest road
(directed-trail semantics, holder-keeps-ties, set-aside on non-holder ties),
largest army, **player-to-player trading**, and immediate victory at 10 VP
from any source.

Player trading uses a bounded offer menu (RL-friendly): the current player
offers 1-2 of one resource for 1 of another (max 3 offers per turn); each
eligible responder accepts or rejects in seat order; the proposer then picks
a partner or cancels. Free-form multi-resource bundles are a deliberate
non-goal for now — the action space stays ~470 discrete actions.

## Testing strategy (the fortress)

1. **Rulebook scenario tests** (`tests/rules_*.rs`) — table-driven cases per
   rules area: building, trading, ports, dev cards, robber, discards, player
   trading, road-building card, victory, phase guards.
2. **Property tests** (`tests/properties.rs`) — random and
   proptest-generated games checked after every action: resource
   conservation, non-negativity, building limits and counter consistency,
   global distance rule, dev-card accounting, winner validity.
3. **Oracles** — a brute-force longest-trail implementation cross-checks the
   fast DFS on live game states. (This caught a real bug inherited from the
   Python engine: its DFS was not direction-aware and could overcount road
   length by pivoting back through a shared vertex.)
4. **Illegal-action fuzzing** — at sampled states, all ~1,700
   parameterizations of every action are fired at the engine: anything
   outside the legal mask must be rejected with zero side effects, and
   everything inside it must execute. Mask and executor can't drift.
5. **Self-goldens** (`tests/self_golden.rs`) — 60 recorded games (~89k
   actions, randomness stored explicitly) replay to field-exact final
   states. This is the drift net for optimization work. Regenerate after
   intentional rule changes:
   `cargo run -p catan-sim --release -- --games 60 --record-golden catan-core/tests/golden/self`

## Rule fixes vs. the legacy Python engine

The Python engine (`engine/`) remains for the visualizer but is no longer
the reference. Fixed here, intentionally diverging:

- Victory is immediate from any source (Python only checked on builds).
- Road Building places as many roads as possible; never deadlocks the game
  (and isn't offered when it would place zero).
- Longest road uses true trail semantics and official award/tie rules.
- The port-type shuffle actually governs trade rates (vestigial in Python).
- Discarding on a 7 is the player's choice, one card at a time (Python
  offered 10 random pre-baked bundles).
- Every action is phase/player/legality-validated; illegal actions are
  rejected atomically.

## Live dashboard

`catan-web` serves a zero-dependency live dashboard at
`http://127.0.0.1:5050/dashboard` that tails a metrics JSONL stream
(`--metrics-file <path>`; schema in `../training/README.md`). It shows
throughput, rolling win rates by seat, game length, turn-cap rate, and —
once training emits `train`/`eval` events — entropy, explained variance,
clip fraction, and fixed-opponent eval curves. A health panel converts the
signals into OK/WATCH/BAD verdicts with plain-language explanations of what
to do. Produce a live stream today with:

```bash
./target/release/catan-sim --games 2000000 --players R,H,R,H --metrics /tmp/m.jsonl
./target/release/catan-web --metrics-file /tmp/m.jsonl   # then open /dashboard
```

## Game replays (CTRP format)

Games can be recorded to a compact binary format (~4.5 bytes/action,
~2-6 KB/game) for later replay and visualization — the source format for
training videos. `GameRecord` in `catan-core/src/replay.rs` records board +
action stream (randomness explicit) + outcome summary; `replay()` rebuilds
the exact game. Tests pin byte roundtrips, replay exactness, and clean
rejection of malformed bytes.

```bash
./target/release/catan-sim --games 100 --players R,H,R,H --record-replays replays/
./target/release/catan-sim --dump-replay replays/game_000000_*.ctrp   # JSON view
```

Training-side artifact storage and checkpoint gating (versioned weights,
promotion rules, replay retention) is specified in `../training/README.md`.

## Running and inspecting the test suite

All commands run from `rust/`:

```bash
cargo test -p catan-core                  # everything: one-line verdict per suite
cargo test -p catan-core -- --nocapture   # also prints the harness summaries:
#   [invariants] 25 games, 61709 actions executed, 61734 invariant sweeps, ...
#   [fuzz] ... 104 states fuzzed (582 legal verified executable,
#               177050 illegal verified rejected)

cargo test -p catan-core --test properties            # just the property/fuzz/oracle harness
cargo test -p catan-core --test rules_player_trading  # one scenario suite
cargo test -p catan-core -- --list                    # list every test by name
cargo test -p catan-core seven_forces                 # run tests matching a name
```

Notes on reading the output:

- Rust tests are silent on success — a passing fuzz run prints nothing
  without `--nocapture`. The `[invariants]` / `[fuzz]` summary lines exist so
  you can see how much was actually checked.
- On failure, the assertion message carries the context (seed, step, action),
  so a red run tells you exactly which game state to reproduce:
  `seed 12 step 448: longest road mismatch for player 2`.
- proptest failures additionally shrink to a minimal failing input and
  persist it in `catan-core/proptest-regressions/` (commit that file — it
  re-runs the exact regression case forever after).
- Coverage: `cargo install cargo-llvm-cov && cargo llvm-cov -p catan-core --html`
  writes an annotated source report to `target/llvm-cov/html/`.

## Commands

```bash
cd rust
cargo build --release

./target/release/catan-sim --games 100000 --players R,R,H,H --seed 42
./target/release/catan-sim --games 1000 --players R,R,R,R --single-thread
```

## Measuring performance

The unit of work is a **step** (one legal-mask generation + one action
execution), not a game — games vary 10x in length by strategy. Every
`catan-sim` run reports `steps/sec` and `ns/step` alongside games/sec.

For the optimization-relevant split, use the step profiler:

```bash
./target/release/catan-sim --games 2000 --players H,H,H,H --profile-steps
```

It reports, single-threaded:
- engine time per step (legal mask + execute) — the "state fulfillment"
  cost and the optimization target;
- agent decision time (replaced by NN inference in RL);
- the engine-only ceiling (steps/sec if the agent were free);
- engine ns/step broken down by turn phase, which points directly at the
  hot phase (it's `main`: 69-83% of engine time).

Numbers on an M-series MacBook (full rules, 20k-game samples, 1 thread,
after the bitmask optimization pass): **87 ns/step** with random players
(11.5M steps/sec) and **221 ns/step** in heuristic games (4.5M steps/sec) —
37M steps/sec aggregate across 10 threads. Caveats: per-step timers add
~100 ns/step to profiler bucket totals (the proportions are what matter),
and single-run wall-clock numbers on macOS swing with P/E-core scheduling —
trust criterion's statistics over one-shot runs.

For reference, the legacy Python engine ran ~5 games/s (~200 ms/game,
roughly 300,000 ns/step) with fewer rules.

## Performance regression protection

Two guards keep the hot path fast permanently:

- **Criterion micro-benchmarks** (`benches/engine.rs`) covering main-phase
  mask generation (typical / rich / dense-board), roll + distribution, road
  builds through the longest-road award path, the 15-road-snake worst case
  (MCTS tail latency, ~1.9 us), and a full game. Workflow:
  `cargo bench -p catan-core --bench engine -- --save-baseline before`, make
  the change, then `-- --baseline before` for statistical pass/fail per
  bench. The baseline lives in `target/criterion/`.
- **Zero-allocation test** (`tests/zero_alloc.rs`): a counting global
  allocator asserts that steady-state steps (mask + agent + execute,
  including 7-roll discards, trades, robber and steals) perform **zero heap
  allocations**. It already caught 15 allocations per ~800 steps when
  introduced; any future per-step `Vec` fails the suite.

The bitmask pass this protects: occupancy/adjacency/road-touch bitboards
make connectivity and distance-rule checks O(1) (mask generation -52% on
rich states), and per-player cached road lengths with adjacency-filtered
recomputes removed whole-board longest-road recomputation from settlement
and road builds (full games -31%, heuristic step time 820 -> 221 ns).
Behavior preserved: the self-golden suite replays 89k recorded actions to
identical final states, and the property/oracle/fuzz harness stayed green
throughout.

## Next (Phase 3): RL environment

Fixed action-id space with legality mask, tensor observation encoder
(perfect- and partial-information modes), batched vectorized `step`, and
PyO3 bindings.
