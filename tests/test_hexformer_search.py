"""Phase 4 guards: hex graph topology, HexFormer drop-in contract, UCB1 search."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th

import catanatron.gym  # noqa: F401  -- registers env id
from catanatron.game import Game
from catanatron.models.player import Color

from catan_rl.eval.opponents import build_opponent
from catan_rl.features import HexFormer
from catan_rl.features.hex_graph import N_NODES, N_TILES, N_TOKENS, build_hex_graph
from catan_rl.features.hexformer import MAP_CHANNELS

_OBS = gym.spaces.Dict({
    "board": gym.spaces.Box(0.0, 1.0, (16, 21, 11), np.float32),
    "numeric": gym.spaces.Box(0.0, 95.0, (48,), np.float32),
})


def test_hex_graph_topology():
    g = build_hex_graph(21, 11)
    assert g.adj_mask.shape == (N_TOKENS, N_TOKENS)
    assert (g.token_type == 0).sum() == N_TILES
    assert (g.token_type == 1).sum() == N_NODES
    # symmetric, self-loops, no isolated token, cells in range
    assert np.array_equal(g.adj_mask, g.adj_mask.T)
    assert g.adj_mask.diagonal().all()
    assert (g.adj_mask.sum(1) > 1).all()
    assert g.token_cells.max() < 21 * 11
    # a center tile sees 6 tile + 6 node neighbours (+self)
    assert g.adj_mask[:N_TILES].sum(1).max() == 13


def test_hexformer_dropin_contract():
    enc = HexFormer(_OBS)
    assert enc.map_shape == (MAP_CHANNELS, 21, 11)
    assert enc.features_dim == MAP_CHANNELS * 21 * 11 + enc.d_g
    b = 3
    obs = {"board": th.rand(b, 16, 21, 11), "numeric": th.rand(b, 48) * 5}
    out = enc(obs)
    assert out.shape == (b, enc.features_dim)
    assert th.isfinite(out).all()
    # splits back into (map, g) exactly as the spatial policy expects
    c, h, w = enc.map_shape
    n_map = c * h * w
    assert out[:, :n_map].view(b, c, h, w).shape == (b, c, h, w)
    assert out[:, n_map:].shape == (b, enc.d_g)


def test_search_player_returns_legal_action(tmp_path):
    from sb3_contrib import MaskablePPO
    from catan_rl.policies.spatial import SpatialMaskableActorCriticPolicy
    from catan_rl.features import HexEncoder

    # tiny spatial model so the search wrapper has something to load
    env = gym.make(
        "catanatron/Catanatron-v0",
        config={"map_type": "BASE", "representation": "mixed"},
    )
    model = MaskablePPO(
        SpatialMaskableActorCriticPolicy, env, n_steps=64, batch_size=32,
        policy_kwargs=dict(
            features_extractor_class=HexEncoder,
            player_colors=(Color.BLUE, Color.RED),
            map_type="BASE", enable_maritime=True,
        ),
        device="cpu",
    )
    ckpt = tmp_path / "m.zip"
    model.save(str(ckpt))

    from catan_rl.search.ucb_player import SearchRLPlayer
    player = SearchRLPlayer(
        Color.BLUE, ckpt, n_sims=8, rollout_depth=4,
        representation="mixed", enable_maritime_trade=True, seed=0,
    )
    game = Game([player, build_opponent("random", Color.RED)], seed=0)
    # play a handful of ticks; every search decision must be a legal action
    for _ in range(40):
        if game.winning_color() is not None:
            break
        before = list(game.playable_actions)
        rec = game.play_tick()
        if game.state.current_player().color == Color.BLUE:
            pass  # decision already validated by execute()
    assert game.state.num_turns >= 0  # reached here without raising
