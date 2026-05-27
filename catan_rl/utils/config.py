"""Base config schema. Phase 0 only uses `EnvConfig`, `EvalConfig`, `WandbConfig`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"
    representation: Literal["vector", "mixed"] = "vector"
    vps_to_win: int = 10


class EvalConfig(BaseModel):
    games: int = 20
    modes: list[Literal["1v1", "4p"]] = Field(default_factory=lambda: ["1v1", "4p"])
    opponents: list[str] = Field(
        default_factory=lambda: ["random", "weighted", "value", "alphabeta1", "alphabeta2"]
    )


class WandbConfig(BaseModel):
    project: str = "catan-rl"
    mode: Literal["online", "offline", "disabled"] = "disabled"
    entity: str | None = None


class BaseConfig(BaseModel):
    seed: int = 0
    env: EnvConfig = Field(default_factory=EnvConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


def load_yaml(path: str | Path) -> BaseConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return BaseConfig.model_validate(data)
