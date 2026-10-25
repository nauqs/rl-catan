"""Config schema for Phase 0 / Phase 1 / Phase 2a."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"
    representation: Literal["vector", "mixed"] = "mixed"
    vps_to_win: int = 10
    # When False (the historical default), MARITIME_TRADE (bank/port 4:1/3:1/2:1)
    # is masked off the agent's action space. Phase 3 sets this True so the agent
    # can trade with the bank like every benchmark bot. Player<->player trades are
    # not in the gym action space at all, so they are unaffected.
    enable_maritime_trade: bool = False


class EvalConfig(BaseModel):
    games: int = 20
    modes: list[Literal["1v1", "4p"]] = Field(default_factory=lambda: ["1v1", "4p"])
    opponents: list[str] = Field(
        default_factory=lambda: ["random", "weighted", "value", "alphabeta1", "alphabeta2"]
    )

    callback_opponents: list[str] = Field(default_factory=lambda: ["random", "weighted", "value"])
    callback_games: int = 20
    callback_every: int = 50_000

    run_final_ladder: bool = True
    final_opponents: list[str] = Field(
        default_factory=lambda: ["random", "weighted", "value", "alphabeta1", "alphabeta2"]
    )
    final_games: int = 500
    final_modes: list[Literal["1v1", "4p"]] = Field(default_factory=lambda: ["1v1"])


class WandbConfig(BaseModel):
    project: str = "catan-rl"
    mode: Literal["online", "offline", "disabled"] = "disabled"
    entity: str | None = None


class PPOConfig(BaseModel):
    gamma: float = 0.999
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 512
    ent_coef: float = 0.01
    learning_rate: float = 3.0e-4
    n_epochs: int = 10
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Linear LR warmup over the first `warmup_steps` env steps (0 = constant LR).
    # Transformer trunks (HexFormer) optimize poorly under the conv-tuned constant
    # 3e-4; warmup + a lower peak tames the large-KL/low-entropy churn.
    warmup_steps: int = 0
    # KL-to-BC anchor: add `kl_coef * KL(pi || pi_ref)` to the policy loss, where
    # pi_ref is a frozen reference policy (the BC seed). Keeps PPO inside a trust
    # region around the competent BC prior so it cannot drift off-manifold and
    # collapse (the v1/v3-lean failure mode). 0 = off. The reference is captured
    # from the run dir's BC seed automatically (see ppo.py); kl_ref_path overrides.
    kl_coef: float = 0.0
    kl_ref_path: str | None = None
    # Adaptive KL: if > 0, kl_coef is auto-tuned each rollout to hold the measured
    # KL(pi||pi_ref) near this target (RLHF-style controller), instead of being a
    # fixed hand-picked strength. Removes the manual coef guessing: too-loose
    # drifts below BC, too-tight freezes at BC. kl_coef is then the *initial* coef.
    kl_target: float = 0.0

    # --- MMD / moving-magnet upgrade (IIG-paper recipe, arXiv 2502.08938) -------
    # The fixed-BC anchor (magnet_mode="fixed") is a floor-protector that caps at
    # BC: exceeding BC means leaving the very manifold it defends. MMD/R-NaD let
    # the magnet MOVE so the policy stays stable step-to-step (no collapse) yet can
    # climb past BC. magnet_mode:
    #   "fixed" -> pi_ref stays the frozen BC seed (original anchor behaviour).
    #   "ema"   -> after every rollout, pi_ref <- (1-magnet_ema)*pi_ref + magnet_ema*pi
    #              (a slowly-moving competent target; allows slow drift, blocks fast).
    #   "reset" -> pi_ref <- copy(pi) every magnet_reset_steps env steps (R-NaD-style
    #              periodic magnet reset to the current policy).
    magnet_mode: Literal["fixed", "ema", "reset"] = "fixed"
    magnet_ema: float = 0.01          # tau for magnet_mode="ema" (per rollout)
    magnet_reset_steps: int = 0       # env steps between resets for magnet_mode="reset"
    # MMD proximal term: add `mmd_kl_coef * KL(pi_new || pi_old)` (backward-KL to the
    # rollout policy, the k3 estimator) to the policy loss. The (1/eta) proximal term
    # in MMD; complements clip as a smoother per-update trust region. 0 = off.
    mmd_kl_coef: float = 0.0
    # Linear-anneal ent_coef from its start value to ent_coef_final over training
    # (None = constant). The IIG recipe starts with high entropy then relaxes.
    ent_coef_final: float | None = None


class RewardConfig(BaseModel):
    """Per-term coefficients for ShapedRewardWrapper.

    Setting any coefficient to 0 disables that term cleanly. Phase-1 defaults
    match Charlesworth; Phase-2a overrides them via configs/curriculum.yaml.
    """
    enabled: bool = True
    anneal_start_step: int = -1     # -1 disables annealing; else linear taper
    anneal_end_step: int = -1

    # Terminal (never annealed)
    win_reward: float = 500.0
    loss_reward: float = 0.0

    # Dense — ANNEALED by dense_factor (VP-generating / placement signals)
    vp_coef: float = 5.0
    city_bonus: float = 2.5
    settlement_bonus: float = 0.0            # flat bonus per BUILD_SETTLEMENT (on top of vp_coef)
    dev_play_reward: float = 5.0
    discard_penalty: float = -0.3
    initial_placement_pip: float = 0.0       # × pip_sum at each initial BUILD_SETTLEMENT
    initial_placement_diversity: float = 0.0 # legacy per-settlement × n_unique_resources
    # Combined opening coverage: lookup on the union of distinct resources across
    # BOTH initial settlements (fired at the 2nd). e.g. {5: 10, 4: 4, 3: -2, 2: -5, 1: -9}.
    placement_coverage: dict[int, float] = Field(default_factory=dict)
    per_turn_penalty: float = 0.0            # applied every env step

    # Legacy robber terms (kept for back-compat; v3 uses the pip-based ones below)
    robber_target_opponent: float = 1.0
    robber_blocks_high_pip: float = 0.0

    # Dense — PERSISTENT (never annealed): efficiency/tactics that get spammed or
    # wasted without a standing signal (roads / maritime / robber placement).
    road_cost: float = 0.0                   # per BUILD_ROAD (small opportunity cost)
    longest_road_delta: float = 0.0          # × increase in longest-road length
    maritime_penalty: float = 0.0            # × (n_gives - 1): 2:1→1, 3:1→2, 4:1→3
    robber_block_opp_pip: float = 0.0        # +coef × tile_pip when an opponent is adjacent
    robber_block_self_pip: float = 0.0       # -coef × tile_pip when WE are adjacent (penalty)
    robber_steal: float = 0.0                # bonus when the move actually steals a card


class OpponentConfig(BaseModel):
    """How opponents are chosen during training."""
    sampler: Literal["single", "mixed", "mixed_with_self", "scheduled", "adaptive", "epsilon"] = "single"
    # --- epsilon curriculum (sampler="epsilon") ---
    # Main non-self rung = eps_base bot playing a random move with prob epsilon.
    # epsilon auto-calibrates (eval callback): wr < target -> easier (raise eps).
    eps_base: str = "value"
    eps_init: float = 0.75
    eps_target: float = 0.5
    eps_lr: float = 0.25                     # eps += eps_lr * (target - winrate), clipped [0,1]
    eps_weight: float = 0.35
    eps_floor_mix: dict[str, float] = Field(default_factory=lambda: {"value": 0.10, "alphabeta1": 0.05})
    eps_games: int = 12                      # calibration games per eval callback
    # --- adaptive curriculum (sampler="adaptive") ---
    # Difficulty ladder (no 'random' — start at weighted). 'self' is always present
    # at `self_weight`; the rest of the budget goes to the <=2 currently-contestable
    # ladder rungs (by recent eval winrate), plus `floor_weight` on the next rung up.
    ladder: list[str] = Field(default_factory=lambda: ["weighted", "value", "alphabeta1", "alphabeta2"])
    self_weight: float = 0.5
    floor_weight: float = 0.05               # small exposure to the next-harder rung
    master_threshold: float = 0.85           # winrate at/above which a rung is "mastered" (drops out)
    best_past_self: bool = True              # self-snapshot tracks the BEST checkpoint, not the latest
    # For sampler="single": just use first key in mix; for "mixed"/"mixed_with_self": weighted.
    mix: dict[str, float] = Field(default_factory=lambda: {"weighted": 1.0})
    # Scheduled-specific: interpolate weights between start_mix and end_mix.
    start_mix: dict[str, float] = Field(default_factory=dict)
    end_mix: dict[str, float] = Field(default_factory=dict)
    schedule_start_step: int = 0
    schedule_end_step: int = 1
    # Self-snapshot params (used by mixed_with_self and scheduled when 'self' is in mix):
    snapshot_every_steps: int = 1_000_000
    self_fallback: str = "weighted"  # used until first snapshot exists


class TrainConfig(BaseModel):
    total_timesteps: int = 10_000_000
    n_envs: int = 8
    device: Literal["cpu", "mps", "cuda", "auto"] = "cpu"
    arch: Literal["boardcnn", "boardcnn_attn", "mlp"] = "boardcnn"
    policy: Literal["flat", "factorized", "spatial"] = "flat"
    # Trunk for the spatial policy: "conv" -> HexEncoder (Phase 3 default),
    # "hexformer" -> adjacency-aware Tile Transformer (Phase 4 ablation).
    trunk: Literal["conv", "hexformer", "xdim"] = "conv"

    # Phase-1 simple-opponent name; if `opponent_cfg` is also set, that wins.
    opponent: Literal["random", "weighted"] = "weighted"
    opponent_cfg: OpponentConfig | None = None

    reward: RewardConfig = Field(default_factory=RewardConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    checkpoint_every: int = 100_000
    # Critic warm-up: freeze the policy, train only the value head for this many
    # env steps before the main RL phase (calibrates V on a BC-seeded policy so
    # PPO's first advantages aren't garbage). 0 = off. Run via scripts/critic_warmup.py.
    critic_warmup_steps: int = 0


class BaseConfig(BaseModel):
    seed: int = 0
    run_name: str = "ppo-baseline"
    env: EnvConfig = Field(default_factory=EnvConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_yaml(path: str | Path) -> BaseConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return BaseConfig.model_validate(data)
