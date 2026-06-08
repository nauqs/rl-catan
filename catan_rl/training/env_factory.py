"""Single source of truth for env construction (training + in-training eval)."""
from __future__ import annotations

from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import catanatron.gym  # noqa: F401 -- registers env id
from stable_baselines3.common.monitor import Monitor

from catanatron import Color

from catan_rl.env.cast_dtype import CastDtypeWrapper
from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.env.shaped_reward import ShapedRewardWrapper
from catan_rl.training.opponents import (
    OpponentSampler,
    SingleOpponent,
)
from catan_rl.utils.config import RewardConfig


class RotatingOpponentWrapper(gym.Wrapper):
    """Mutates `env.unwrapped.enemies` + `players` on every reset() so the
    underlying Catanatron `Game` is rebuilt with the sampled opponent.

    Sits at the innermost wrapper position (right above the gym env) so the
    mutation is visible to the wrapped env's reset(). All other wrappers
    above only see whatever observation/info pair propagates back.
    """

    def __init__(self, env: gym.Env, sampler: OpponentSampler, seed: int = 0):
        super().__init__(env)
        self.sampler = sampler
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        factory = self.sampler.sample(self._rng)
        opp = factory(Color.RED)
        u = self.env.unwrapped
        u.enemies = [opp]
        u.players = [u.p0, opp]
        return self.env.reset(**kwargs)

    # ---- passthroughs for callbacks (AnnealingCallback, OpponentScheduleCallback) ----
    def set_opponent_step(self, step: int) -> None:
        """Forward the current env-step count to the sampler (scheduled samplers
        use it to interpolate between start_mix and end_mix)."""
        self.sampler.set_step(step)

    def set_opponent_winrates(self, winrates: dict) -> None:
        """Forward recent eval winrates to the sampler (adaptive curriculum)."""
        fn = getattr(self.sampler, "set_winrates", None)
        if callable(fn):
            fn(winrates)

    def set_opponent_epsilon(self, eps: float) -> None:
        """Forward the auto-calibrated epsilon to the sampler (epsilon curriculum)."""
        fn = getattr(self.sampler, "set_epsilon", None)
        if callable(fn):
            fn(eps)

    def get_opponent_mix(self) -> dict | None:
        """Current interpolated opponent weights, for W&B logging. Returns None
        for samplers without a schedule (the mix is constant / uninformative)."""
        fn = getattr(self.sampler, "current_mix", None)
        return fn() if callable(fn) else None


def make_env(
    seed: int,
    *,
    map_type: str = "BASE",
    representation: Literal["vector", "mixed"] = "mixed",
    sampler: Optional[OpponentSampler] = None,
    reward_config: Optional[RewardConfig] = None,
    # Phase-1 compat: if `sampler` is None, fall back to the old `opponent` string.
    opponent: str = "weighted",
    reward_shaping: bool = True,
    enable_maritime_trade: bool = False,
) -> gym.Env:
    """Wrapped Catanatron 1v1 env: Rotate -> CastDtype -> NoTrade -> Shaped -> Monitor."""
    if sampler is None:
        sampler = SingleOpponent(opponent)

    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": map_type,
            "representation": representation,
            # initial enemy is a placeholder; RotatingOpponentWrapper replaces it on reset()
            "enemies": [sampler.sample(np.random.default_rng(seed))(Color.RED)],
        },
    )
    env = RotatingOpponentWrapper(env, sampler, seed=seed)
    env = CastDtypeWrapper(env)
    env = NoTradeWrapper(env, strip_maritime=not enable_maritime_trade)
    env = ShapedRewardWrapper(env, enabled=reward_shaping, config=reward_config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env
