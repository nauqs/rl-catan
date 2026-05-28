"""Architecture variants: each registers, instantiates, forward-passes correctly."""
from __future__ import annotations

import gymnasium as gym
import torch

import catanatron.gym  # noqa: F401

from catan_rl.features import EXTRACTORS, BoardCNN, BoardCNNAttn, MLPExtractor


def _obs_space():
    env = gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    space = env.observation_space
    env.close()
    return space


def test_registry_keys() -> None:
    assert set(EXTRACTORS) == {"boardcnn", "boardcnn_attn", "mlp"}
    assert EXTRACTORS["boardcnn"] is BoardCNN
    assert EXTRACTORS["boardcnn_attn"] is BoardCNNAttn
    assert EXTRACTORS["mlp"] is MLPExtractor


def _make_batch(space):
    obs = space.sample()
    return {
        "board": torch.as_tensor(obs["board"]).unsqueeze(0).float(),
        "numeric": torch.as_tensor(obs["numeric"]).unsqueeze(0).float(),
    }


def test_boardcnn_forward_and_params() -> None:
    space = _obs_space()
    m = BoardCNN(space, features_dim=512)
    out = m(_make_batch(space))
    assert out.shape == (1, 512)
    n = sum(p.numel() for p in m.parameters())
    assert 200_000 < n < 2_000_000, f"boardcnn unexpected param count: {n}"


def test_boardcnn_attn_forward_and_params() -> None:
    space = _obs_space()
    m = BoardCNNAttn(space, features_dim=512)
    out = m(_make_batch(space))
    assert out.shape == (1, 512)
    n = sum(p.numel() for p in m.parameters())
    assert 200_000 < n < 5_000_000, f"boardcnn_attn unexpected param count: {n}"


def test_mlp_forward_and_params() -> None:
    space = _obs_space()
    m = MLPExtractor(space, features_dim=512)
    out = m(_make_batch(space))
    assert out.shape == (1, 512)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 20_000_000, f"mlp unexpected param count: {n}"
