"""Save freshly-initialized MaskablePPO checkpoints (vector + mixed) and play full games."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from sb3_contrib import MaskablePPO

import catanatron.gym  # noqa: F401
from catanatron import Color, Game, RandomPlayer
from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.features.board_cnn import BoardCNN
from catan_rl.players.rl_player import RLPlayer


@pytest.fixture(scope="module")
def vector_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    env = NoTradeWrapper(
        gym.make("catanatron/Catanatron-v0", config={"representation": "vector"})
    )
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    path = tmp_path_factory.mktemp("ckpt-v") / "smoke.zip"
    model.save(str(path))
    env.close()
    return path


@pytest.fixture(scope="module")
def mixed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    env = NoTradeWrapper(
        gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    )
    policy_kwargs = dict(
        features_extractor_class=BoardCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = MaskablePPO(
        "MultiInputPolicy", env, n_steps=64, batch_size=32,
        policy_kwargs=policy_kwargs, verbose=0,
    )
    path = tmp_path_factory.mktemp("ckpt-m") / "smoke.zip"
    model.save(str(path))
    env.close()
    return path


def test_rl_player_vector(vector_checkpoint: Path) -> None:
    game = Game(
        [RLPlayer(Color.BLUE, vector_checkpoint, representation="vector"),
         RandomPlayer(Color.RED)],
        seed=0,
    )
    game.play()
    assert game.winning_color() is not None or game.state.num_turns >= 1


def test_rl_player_mixed(mixed_checkpoint: Path) -> None:
    game = Game(
        [RLPlayer(Color.BLUE, mixed_checkpoint, representation="mixed"),
         RandomPlayer(Color.RED)],
        seed=0,
    )
    game.play()
    assert game.winning_color() is not None or game.state.num_turns >= 1
