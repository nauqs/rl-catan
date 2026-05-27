"""Config schema. Phase 0 used `EnvConfig`, `EvalConfig`, `WandbConfig`; Phase 1 adds `TrainConfig`."""
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
    # Phase-0 default ladder fields (used by `catan_rl.eval`):
    games: int = 20
    modes: list[Literal["1v1", "4p"]] = Field(default_factory=lambda: ["1v1", "4p"])
    opponents: list[str] = Field(
        default_factory=lambda: ["random", "weighted", "value", "alphabeta1", "alphabeta2"]
    )

    # Phase-1 in-training callback (cheap ladder run during PPO training):
    callback_opponents: list[str] = Field(default_factory=lambda: ["random", "weighted", "value"])
    callback_games: int = 20
    callback_every: int = 50_000

    # Phase-1 end-of-training full ladder:
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
    enabled: bool = True
    anneal_start_step: int = -1
    anneal_end_step: int = -1


class TrainConfig(BaseModel):
    total_timesteps: int = 10_000_000
    n_envs: int = 8
    device: Literal["cpu", "mps", "cuda", "auto"] = "cpu"
    opponent: Literal["random", "weighted"] = "weighted"
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
