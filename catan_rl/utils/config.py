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


class RewardConfig(BaseModel):
    """Per-term coefficients for ShapedRewardWrapper.

    Setting any coefficient to 0 disables that term cleanly. Phase-1 defaults
    match Charlesworth; Phase-2a overrides them via configs/phase2a.yaml.
    """
    enabled: bool = True
    anneal_start_step: int = -1     # -1 disables annealing; else linear taper
    anneal_end_step: int = -1

    # Terminal (never annealed)
    win_reward: float = 500.0
    loss_reward: float = 0.0

    # Dense (annealed by dense_factor)
    vp_coef: float = 5.0
    city_bonus: float = 2.5
    dev_play_reward: float = 5.0
    robber_target_opponent: float = 1.0      # only when MOVE_ROBBER targets opponent
    robber_blocks_high_pip: float = 0.0      # bonus × pip_sum × (1 if opponent adjacent else 0)
    discard_penalty: float = -0.3
    initial_placement_pip: float = 0.0       # × pip_sum at each initial BUILD_SETTLEMENT
    initial_placement_diversity: float = 0.0 # × n_unique_resources at initial BUILD_SETTLEMENT
    per_turn_penalty: float = 0.0            # applied every env step


class OpponentConfig(BaseModel):
    """How opponents are chosen during training."""
    sampler: Literal["single", "mixed", "mixed_with_self", "scheduled"] = "single"
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
    policy: Literal["flat", "factorized"] = "flat"

    # Phase-1 simple-opponent name; if `opponent_cfg` is also set, that wins.
    opponent: Literal["random", "weighted"] = "weighted"
    opponent_cfg: OpponentConfig | None = None

    reward: RewardConfig = Field(default_factory=RewardConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    checkpoint_every: int = 100_000


class BaseConfig(BaseModel):
    seed: int = 0
    run_name: str = "phase1-baseline"
    env: EnvConfig = Field(default_factory=EnvConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_yaml(path: str | Path) -> BaseConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return BaseConfig.model_validate(data)
