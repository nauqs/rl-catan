"""BoardCNN: instantiates against the live mixed-obs space, forward pass shape OK, ~600k params."""
from __future__ import annotations

import gymnasium as gym
import torch

import catanatron.gym  # noqa: F401

from catan_rl.features.board_cnn import BoardCNN


def test_board_cnn_forward_and_size() -> None:
    env = gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    extractor = BoardCNN(env.observation_space, features_dim=512)
    obs, _ = env.reset(seed=0)
    batch = {
        "board": torch.as_tensor(obs["board"]).unsqueeze(0).float(),
        "numeric": torch.as_tensor(obs["numeric"]).unsqueeze(0).float(),
    }
    out = extractor(batch)
    assert out.shape == (1, 512)
    n_params = sum(p.numel() for p in extractor.parameters())
    assert 200_000 < n_params < 2_000_000, f"unexpected param count: {n_params}"
    env.close()
