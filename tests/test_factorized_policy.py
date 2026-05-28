"""FactorizedActionNet + FactorizedMaskableActorCriticPolicy: meta correctness,
forward shape, logit-decomposition math, end-to-end checkpoint round-trip."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO

import catanatron.gym  # noqa: F401  -- registers env id
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType
from catanatron.models.player import Color

from catan_rl.env.no_trade import NoTradeWrapper
from catan_rl.policies.factorized import (
    FactorizedActionNet,
    FactorizedMaskableActorCriticPolicy,
    build_action_meta,
)
from catan_rl.training.env_factory import make_env


_COLORS = (Color.BLUE, Color.RED)


# ---------- meta correctness ----------

def test_action_meta_shape_and_counts() -> None:
    meta = build_action_meta(_COLORS, "BASE")
    assert meta["n_flat"] == 332
    assert meta["flat_to_type"].shape == (332,)
    assert meta["flat_to_param"].shape == (332,)
    assert meta["n_types"] == len(list(ActionType))
    # Heads we expect to be allocated:
    assert "node" in meta["head_sizes"], "BUILD_SETTLEMENT/BUILD_CITY should share 'node'"
    assert meta["head_sizes"]["node"] == 54


def test_meta_flat_to_type_matches_action_array() -> None:
    """For every flat index, the recovered ActionType matches the action array."""
    meta = build_action_meta(_COLORS, "BASE")
    array = get_action_array(_COLORS, "BASE")
    all_types = list(ActionType)
    for flat_idx, (atype, _) in enumerate(array):
        recovered = all_types[int(meta["flat_to_type"][flat_idx])]
        assert recovered == atype, f"flat={flat_idx}: meta={recovered}, actual={atype}"


def test_settlement_city_share_node_head_indices() -> None:
    """For each node_id, BUILD_SETTLEMENT(node_id) and BUILD_CITY(node_id) must
    map to the same param index (so the shared head sees consistent semantics)."""
    meta = build_action_meta(_COLORS, "BASE")
    array = get_action_array(_COLORS, "BASE")
    set_param = {}
    cty_param = {}
    for flat_idx, (atype, value) in enumerate(array):
        if atype == ActionType.BUILD_SETTLEMENT:
            set_param[int(value)] = int(meta["flat_to_param"][flat_idx])
        elif atype == ActionType.BUILD_CITY:
            cty_param[int(value)] = int(meta["flat_to_param"][flat_idx])
    assert set_param == cty_param, "settlement/city must share node-indexed params"


def test_maritime_trade_has_no_head() -> None:
    """MARITIME_TRADE is always masked; no param head should be allocated for it."""
    meta = build_action_meta(_COLORS, "BASE")
    # Find MARITIME_TRADE type id
    all_types = list(ActionType)
    mt_id = all_types.index(ActionType.MARITIME_TRADE)
    assert meta["type_to_head"][mt_id] is None
    # And no head named "head_MARITIME_TRADE"
    assert "head_MARITIME_TRADE" not in meta["head_sizes"]


# ---------- forward + logit decomposition ----------

def test_forward_shape() -> None:
    meta = build_action_meta(_COLORS, "BASE")
    net = FactorizedActionNet(features_dim=64, meta=meta)
    out = net(th.randn(3, 64))
    assert out.shape == (3, 332)
    assert th.isfinite(out).all()


def test_logit_decomposition_holds() -> None:
    """For a known features vector, manually compute flat_logit[a] = type + param
    and compare to the net's output."""
    meta = build_action_meta(_COLORS, "BASE")
    net = FactorizedActionNet(features_dim=32, meta=meta)
    net.eval()
    feats = th.randn(1, 32)

    with th.no_grad():
        type_logits = net.type_head(feats).squeeze(0)               # (n_types,)
        param_logits = {
            name: net.param_heads[name](feats).squeeze(0)            # (head_size,)
            for name in meta["head_sizes"]
        }
        out = net(feats).squeeze(0)                                  # (n_flat,)

    # Spot-check a handful of flat indices: one trivial type, one shared-head, one unshared.
    array = get_action_array(_COLORS, "BASE")
    all_types = list(ActionType)
    type_to_head = meta["type_to_head"]
    samples = [0, 100, 250, 331]   # whatever's in range
    for flat_idx in samples:
        atype, value = array[flat_idx]
        tid = all_types.index(atype)
        expected = type_logits[tid].item()
        head_name = type_to_head[tid]
        if head_name is not None:
            param_idx = int(meta["flat_to_param"][flat_idx])
            expected += param_logits[head_name][param_idx].item()
        assert out[flat_idx].item() == pytest.approx(expected, abs=1e-5)


# ---------- end-to-end training smoke + RLPlayer round-trip ----------

@pytest.fixture(scope="module")
def factorized_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    env = NoTradeWrapper(
        gym.make("catanatron/Catanatron-v0", config={"representation": "mixed"})
    )
    from catan_rl.features.board_cnn import BoardCNN
    policy_kwargs = dict(
        features_extractor_class=BoardCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        player_colors=_COLORS,
        map_type="BASE",
    )
    model = MaskablePPO(
        FactorizedMaskableActorCriticPolicy,
        env,
        n_steps=64,
        batch_size=32,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    # One mini-update so the optimizer + buffers actually exercise.
    model.learn(total_timesteps=128, progress_bar=False)
    path = tmp_path_factory.mktemp("ckpt-factorized") / "smoke.zip"
    model.save(str(path))
    env.close()
    return path


def test_factorized_rl_player_plays_full_game(factorized_checkpoint: Path) -> None:
    """Load the factorized checkpoint via RLPlayer and play a full game.
    Confirms the masked predict() works end-to-end."""
    from catan_rl.players.rl_player import RLPlayer
    from catanatron import Color, Game, RandomPlayer
    game = Game(
        [RLPlayer(Color.BLUE, factorized_checkpoint, representation="mixed"),
         RandomPlayer(Color.RED)],
        seed=0,
    )
    game.play()
    assert game.winning_color() is not None or game.state.num_turns >= 1
