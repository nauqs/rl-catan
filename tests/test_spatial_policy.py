"""Spatial pointer policy: meta correctness, forward shape, positional
sensitivity, separate settle/city scorers, end-to-end checkpoint round-trip."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
import torch as th
from sb3_contrib import MaskablePPO

import catanatron.gym  # noqa: F401  -- registers env id
from catanatron.gym.board_tensor_features import get_node_and_edge_maps
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType
from catanatron.models.player import Color

from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.features import HexEncoder
from catan_rl.policies.spatial import (
    SpatialActionHead,
    SpatialMaskableActorCriticPolicy,
    build_spatial_action_meta,
)

_COLORS = (Color.BLUE, Color.RED)
_MAP_SHAPE = (64, 21, 11)
_D_G = 256


# ---------- meta correctness ----------

def test_meta_counts_and_kinds() -> None:
    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    assert meta["n_flat"] == 332
    counts = {k: len(v) for k, v in meta["spatial"].items()}
    assert counts == {"node_settle": 54, "node_city": 54, "edge": 72, "tile": 57}
    assert {k: v["size"] for k, v in meta["globals"].items()} == {
        "maritime": 60, "year_of_plenty": 20, "monopoly": 5, "discard": 5
    }
    assert meta["n_victims"] == 3  # BLUE / RED / None in 1v1


def test_every_flat_action_classified_once() -> None:
    """Every flat index appears in exactly one place (spatial / global) or is a
    trivial/masked type (type-bias only), and flat_to_type matches the array."""
    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    array = get_action_array(_COLORS, "BASE")
    all_types = list(ActionType)

    placed = set()
    for entries in meta["spatial"].values():
        for e in entries:
            placed.add(e[0])
    for spec in meta["globals"].values():
        for flat_idx, _ in spec["entries"]:
            placed.add(flat_idx)

    trivial_types = {
        ActionType.ROLL, ActionType.END_TURN, ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_ROAD_BUILDING,
    }
    for flat_idx, (atype, _) in enumerate(array):
        assert all_types[int(meta["flat_to_type"][flat_idx])] == atype
        if atype in trivial_types:
            assert flat_idx not in placed
        else:
            assert flat_idx in placed


def test_settlement_city_cells_resolve() -> None:
    """Every settlement/city node_id maps to a valid (i,j) on the 21x11 grid."""
    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    for kind in ("node_settle", "node_city"):
        for _flat, i, j in meta["spatial"][kind]:
            assert 0 <= i <= 20 and 0 <= j <= 10


def test_maritime_present_iff_enabled() -> None:
    on = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    off = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=False)
    assert "maritime" in on["globals"]
    assert "maritime" not in off["globals"]


# ---------- forward shape + finiteness ----------

def test_head_forward_shape() -> None:
    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    head = SpatialActionHead(meta, map_shape=_MAP_SHAPE, d_g=_D_G)
    out = head(th.randn(3, *_MAP_SHAPE), th.randn(3, _D_G))
    assert out.shape == (3, 332)
    assert th.isfinite(out).all()


# ---------- positional sensitivity (the core fix) ----------

def _settlement_flat_for_node(node_id: int) -> int:
    array = get_action_array(_COLORS, "BASE")
    for flat_idx, (atype, value) in enumerate(array):
        if atype == ActionType.BUILD_SETTLEMENT and int(value) == node_id:
            return flat_idx
    raise AssertionError(f"no settlement action for node {node_id}")


def test_positional_sensitivity() -> None:
    """Perturbing the board map at one node's cell shifts *that* node's
    settlement logit but not a distant node's — proves the gather/position fix."""
    node_map, _ = get_node_and_edge_maps()
    # Pick two nodes whose grid cells differ.
    n_a, n_b = 0, 53
    (ia, ja), (ib, jb) = node_map[n_a], node_map[n_b]
    assert (ia, ja) != (ib, jb)

    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    head = SpatialActionHead(meta, map_shape=_MAP_SHAPE, d_g=_D_G).eval()
    fa, fb = _settlement_flat_for_node(n_a), _settlement_flat_for_node(n_b)

    th.manual_seed(0)
    feat_map = th.randn(1, *_MAP_SHAPE)
    g = th.randn(1, _D_G)
    with th.no_grad():
        base = head(feat_map, g).squeeze(0)
        perturbed_map = feat_map.clone()
        perturbed_map[0, :, ia, ja] += 10.0  # only node A's cell
        new = head(perturbed_map, g).squeeze(0)

    assert (new[fa] - base[fa]).abs() > 1e-3, "node A's logit should move"
    assert (new[fb] - base[fb]).abs() < 1e-5, "distant node B's logit should not move"


def test_settle_city_scorers_independent() -> None:
    """Settlement and city share a node cell but use separate scorers, so their
    logits respond to the cell features independently."""
    node_map, _ = get_node_and_edge_maps()
    node_id = 10
    i, j = node_map[node_id]
    array = get_action_array(_COLORS, "BASE")
    f_set = next(k for k, (t, v) in enumerate(array)
                 if t == ActionType.BUILD_SETTLEMENT and int(v) == node_id)
    f_cty = next(k for k, (t, v) in enumerate(array)
                 if t == ActionType.BUILD_CITY and int(v) == node_id)

    meta = build_spatial_action_meta(_COLORS, "BASE", enable_maritime=True)
    head = SpatialActionHead(meta, map_shape=_MAP_SHAPE, d_g=_D_G).eval()
    th.manual_seed(1)
    feat_map = th.randn(1, *_MAP_SHAPE)
    g = th.randn(1, _D_G)
    with th.no_grad():
        base = head(feat_map, g).squeeze(0)
        feat_map[0, :, i, j] += 5.0
        new = head(feat_map, g).squeeze(0)
    d_set = (new[f_set] - base[f_set]).item()
    d_cty = (new[f_cty] - base[f_cty]).item()
    # Both move (both read the cell) but by different amounts (separate scorers).
    assert abs(d_set) > 1e-3 and abs(d_cty) > 1e-3
    assert abs(d_set - d_cty) > 1e-4


# ---------- end-to-end training smoke + RLPlayer round-trip ----------

@pytest.fixture(scope="module")
def spatial_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    env = NoTradeWrapper(
        gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"}),
        strip_maritime=False,
    )
    policy_kwargs = dict(
        features_extractor_class=HexEncoder,
        player_colors=_COLORS,
        map_type="BASE",
        enable_maritime=True,
    )
    model = MaskablePPO(
        SpatialMaskableActorCriticPolicy,
        env,
        n_steps=64,
        batch_size=32,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    model.learn(total_timesteps=128, progress_bar=False)
    path = tmp_path_factory.mktemp("ckpt-spatial") / "smoke.zip"
    model.save(str(path))
    env.close()
    return path


def test_spatial_rl_player_plays_full_game(spatial_checkpoint: Path) -> None:
    from catan_rl.players.rl_player import RLPlayer
    from catanatron import Color, Game, RandomPlayer
    game = Game(
        [RLPlayer(Color.BLUE, spatial_checkpoint, representation="mixed",
                  enable_maritime_trade=True),
         RandomPlayer(Color.RED)],
        seed=0,
    )
    game.play()
    assert game.winning_color() is not None or game.state.num_turns >= 1
