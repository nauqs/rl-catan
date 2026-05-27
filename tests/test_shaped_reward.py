"""ShapedRewardWrapper: terminal +500/0, dense terms accumulate, annealing scales dense only."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import catanatron.gym  # noqa: F401

from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.env.shaped_reward import ShapedRewardWrapper


def _wrap() -> ShapedRewardWrapper:
    env = gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    return ShapedRewardWrapper(NoTradeWrapper(env))


def _play_random(env, seed: int) -> tuple[float, bool]:
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        valid = info["valid_actions"]
        action = int(rng.choice(valid))
        obs, r, terminated, truncated, info = env.step(action)
        total += r
    won = env.unwrapped.game.winning_color() == env.unwrapped.p0.color
    return total, won


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shaped_rewards_finite_and_terminal_dominant(seed: int) -> None:
    env = _wrap()
    total, won = _play_random(env, seed)
    assert np.isfinite(total)
    if won:
        assert total >= 500.0, "winner should accumulate at least the +500 terminal"
    else:
        # Loser gets no terminal, but dense terms (incl. -0.3 per discard) can be any sign.
        # Just bound the magnitude loosely.
        assert -50.0 < total < 200.0
    env.close()


def test_annealing_factor_scales_dense_only() -> None:
    env = _wrap()
    env.set_dense_factor(0.0)
    total, won = _play_random(env, seed=0)
    if won:
        assert total == pytest.approx(500.0, abs=1e-6)
    else:
        assert total == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_disabled_passthrough_returns_catanatron_default() -> None:
    env = gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    env = ShapedRewardWrapper(NoTradeWrapper(env), enabled=False)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        action = int(rng.choice(info["valid_actions"]))
        obs, r, terminated, truncated, info = env.step(action)
        total += r
    # Catanatron's simple_reward returns +1 / 0 / -1 only at terminal step.
    assert total in (-1.0, 0.0, 1.0)
    env.close()
