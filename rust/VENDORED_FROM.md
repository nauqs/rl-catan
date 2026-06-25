# Vendored Rust engine — attribution

The `catan-core` (rules engine), `catan-env` (obs/action codec, VecEnv, AlphaBot),
`catan-py` (PyO3 bindings), `catan-sim`, and `catan-web` crates here are vendored
from https://github.com/Eli6th/catan-rl (MIT — see `UPSTREAM_LICENSE`), 2026-06-25.

We use this fast engine as a drop-in simulator under our existing Python training
stack (PPO / wandb / metrics). Modifications are noted in git history.
