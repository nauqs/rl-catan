"""Save a freshly-initialized MaskablePPO and play full games with RLPlayer."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from sb3_contrib import MaskablePPO

import catanatron.gym  # noqa: F401
from catanatron import Color, Game, RandomPlayer
from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.players.rl_player import RLPlayer


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    env = NoTradeWrapper(gym.make("catanatron/Catanatron-v0"))
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    path = tmp_path_factory.mktemp("ckpt") / "smoke.zip"
    model.save(str(path))
    env.close()
    return path


def test_rl_player_finishes_games(checkpoint: Path) -> None:
    for seed in range(2):
        game = Game(
            [
                RLPlayer(Color.BLUE, checkpoint),
                RandomPlayer(Color.RED),
            ],
            seed=seed,
        )
        game.play()
        assert game.winning_color() is not None or game.state.num_turns >= 1
