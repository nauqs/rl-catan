"""ShapedRewardWrapper: defaults, annealing, disabled passthrough, plus Phase-2a new terms."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import catanatron.gym  # noqa: F401

from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.env.shaped_reward import ShapedRewardWrapper
from catan_rl.utils.config import RewardConfig


def _wrap(config: RewardConfig | None = None) -> ShapedRewardWrapper:
    env = gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    return ShapedRewardWrapper(NoTradeWrapper(env), config=config)


def _play_random(env, seed: int) -> tuple[float, bool]:
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.choice(info["valid_actions"]))
        obs, r, terminated, truncated, info = env.step(action)
        total += r
    won = env.unwrapped.game.winning_color() == env.unwrapped.p0.color
    return total, won


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_phase1_defaults_finite_and_terminal_dominant(seed: int) -> None:
    env = _wrap()  # default config = Phase-1 values
    total, won = _play_random(env, seed)
    assert np.isfinite(total)
    if won:
        # Winner gets the +500 terminal on top of (non-negative-ish) dense terms.
        assert total >= 500.0
    # Loser's total is pure dense reward (loss_reward defaults to 0); its
    # magnitude depends on game length and is not meaningfully bounded — only
    # assert finiteness, which is the property under test.
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
    env = _wrap(RewardConfig(enabled=False))
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        action = int(rng.choice(info["valid_actions"]))
        obs, r, terminated, truncated, info = env.step(action)
        total += r
    assert total in (-1.0, 0.0, 1.0)
    env.close()


# -------- Phase-2a-specific term behaviour ----------

def _zeroed_reward() -> RewardConfig:
    """All coefficients set to 0; nothing should ever fire (terminal=0 too)."""
    return RewardConfig(
        win_reward=0.0, loss_reward=0.0,
        vp_coef=0.0, city_bonus=0.0, dev_play_reward=0.0,
        robber_target_opponent=0.0, robber_blocks_high_pip=0.0,
        discard_penalty=0.0,
        initial_placement_pip=0.0, initial_placement_diversity=0.0,
        per_turn_penalty=0.0,
    )


def test_all_zero_yields_zero_total_reward() -> None:
    env = _wrap(_zeroed_reward())
    total, _ = _play_random(env, seed=0)
    assert total == pytest.approx(0.0, abs=1e-9)
    env.close()


def test_per_turn_penalty_only_accumulates_per_step() -> None:
    cfg = _zeroed_reward().model_copy(update={"per_turn_penalty": -0.1})
    env = _wrap(cfg)
    obs, info = env.reset(seed=7)
    rng = np.random.default_rng(7)
    n_steps = 0
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(int(rng.choice(info["valid_actions"])))
        n_steps += 1
        total += r
    expected = -0.1 * n_steps
    assert total == pytest.approx(expected, abs=1e-6)
    env.close()


def test_loss_reward_fires_only_on_terminal_loss() -> None:
    cfg = _zeroed_reward().model_copy(update={"win_reward": 0.0, "loss_reward": -123.0})
    env = _wrap(cfg)
    rng = np.random.default_rng(0)
    # Play several games; whichever the agent loses should give exactly -123 (others 0).
    for s in range(5):
        total, won = _play_random(env, seed=s + 100)
        if won:
            assert total == pytest.approx(0.0, abs=1e-6)
        else:
            assert total == pytest.approx(-123.0, abs=1e-6)
    env.close()


def test_win_reward_fires_only_on_terminal_win() -> None:
    cfg = _zeroed_reward().model_copy(update={"win_reward": 77.0, "loss_reward": 0.0})
    env = _wrap(cfg)
    for s in range(5):
        total, won = _play_random(env, seed=s + 200)
        if won:
            assert total == pytest.approx(77.0, abs=1e-6)
        else:
            assert total == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_vp_delta_only_telescopes_to_5_times_final_vp() -> None:
    from catanatron.state_functions import get_actual_victory_points
    cfg = _zeroed_reward().model_copy(update={"vp_coef": 5.0})
    env = _wrap(cfg)
    for s in range(3):
        total, _ = _play_random(env, seed=s + 300)
        final_vp = get_actual_victory_points(env.unwrapped.game.state, env.unwrapped.p0.color)
        assert total == pytest.approx(5.0 * final_vp, abs=1e-6)
    env.close()
