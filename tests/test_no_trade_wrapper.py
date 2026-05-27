"""Verify NoTradeWrapper never surfaces a trade action across many rollouts."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import catanatron.gym  # noqa: F401  -- registers env id
from catan_rl.env.no_trade import NoTradeWrapper, trade_action_indices


@pytest.mark.parametrize("seed", range(5))
def test_no_trade_in_rollouts(seed: int) -> None:
    env = NoTradeWrapper(gym.make("catanatron/Catanatron-v0"))
    trade = trade_action_indices(env)
    rng = np.random.default_rng(seed)

    games = 0
    while games < 4:
        obs, info = env.reset(seed=seed + games * 17)
        assert not (set(info["valid_actions"]) & trade)
        mask = env.action_masks()
        assert not any(mask[i] for i in trade)

        while True:
            valid = info["valid_actions"]
            assert valid, "no non-trade action available — wrapper over-masked"
            action = int(rng.choice(valid))
            assert action not in trade
            obs, _, terminated, truncated, info = env.step(action)
            assert not (set(info["valid_actions"]) & trade)
            if terminated or truncated:
                break
        games += 1
    env.close()
