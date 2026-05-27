"""Single source of truth for env construction (training + in-training eval)."""
from __future__ import annotations

from typing import Callable, Literal

import gymnasium as gym
import catanatron.gym  # noqa: F401 -- registers env id
from stable_baselines3.common.monitor import Monitor

from catanatron import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.env.cast_dtype import CastDtypeWrapper
from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.env.shaped_reward import ShapedRewardWrapper

OpponentName = Literal["random", "weighted"]


def _opponent_factory(name: OpponentName) -> Callable[[Color], Player]:
    if name == "random":
        return lambda c: RandomPlayer(c)
    if name == "weighted":
        return lambda c: WeightedRandomPlayer(c)
    raise ValueError(f"unknown opponent for training: {name}")


def make_env(
    seed: int,
    *,
    map_type: str = "BASE",
    representation: Literal["vector", "mixed"] = "mixed",
    opponent: OpponentName = "weighted",
    reward_shaping: bool = True,
) -> gym.Env:
    """Wrapped Catanatron 1v1 env: NoTrade + (optional) shaped reward + Monitor."""
    opp_factory = _opponent_factory(opponent)
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": map_type,
            "representation": representation,
            "enemies": [opp_factory(Color.RED)],
        },
    )
    env = CastDtypeWrapper(env)
    env = NoTradeWrapper(env)
    env = ShapedRewardWrapper(env, enabled=reward_shaping)
    env = Monitor(env)
    env.reset(seed=seed)
    return env
