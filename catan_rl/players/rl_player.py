"""Catanatron Player that wraps a trained MaskablePPO checkpoint."""
from __future__ import annotations

import contextlib
import io
import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO


@contextlib.contextmanager
def _force_cpu_deserialize():
    """Force every torch storage to deserialize onto CPU during a load.

    Snapshots saved from an MPS-trained model embed MPS-device tensors (both in
    the policy state_dict and in SB3's cloudpickled `data` section, which ignores
    the `device=` load arg). A forkserver env worker cannot initialize Metal, so
    deserializing an MPS storage there aborts the whole worker process
    (`storage.mps()` -> Fatal Python error: Aborted) -> the parent EOFs on the
    pipe. Pinning `_load_from_bytes` to map_location='cpu' makes the worker load
    any snapshot (MPS- or CPU-saved) without ever touching Metal. pickle resolves
    the reducer by module+qualname at load time, so patching the attribute here
    is picked up by the unpickler.
    """
    orig = th.storage._load_from_bytes

    def _cpu_load_from_bytes(b):
        return th.load(io.BytesIO(b), map_location="cpu")

    th.storage._load_from_bytes = _cpu_load_from_bytes
    try:
        yield
    finally:
        th.storage._load_from_bytes = orig

from catanatron import Player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.board_tensor_features import (
    create_board_tensor,
    is_graph_feature,
)
from catanatron.gym.envs.action_space import (
    from_action_space,
    get_action_array,
    to_action_space,
)
from catanatron.models.player import Color

from catan_rl.env.no_trade import TRADE_ACTION_TYPES


class RLPlayer(Player):
    """Acts in a Catanatron game using a MaskablePPO policy.

    Rebuilds the observation in the same shape the gym env produced
    (`vector` or `mixed`), builds an action mask from `playable_actions`
    (trade indices masked off to match training), runs
    `model.predict(deterministic=True)`, and decodes the chosen index back
    to a Catanatron `Action`.
    """

    def __init__(
        self,
        color: Color,
        model_path: str | Path,
        *,
        map_type: str = "BASE",
        representation: Literal["vector", "mixed"] = "mixed",
        deterministic: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
        enable_maritime_trade: bool = False,
    ):
        super().__init__(color)
        # device defaults to "cpu": opponents run inside forkserver env workers
        # where MPS/Metal is unavailable. Force CPU deserialization so an
        # MPS-saved snapshot can't abort the worker (see _force_cpu_deserialize).
        with _force_cpu_deserialize():
            self._model = MaskablePPO.load(str(model_path), device=device)
        self._map_type = map_type
        self._representation = representation
        self._deterministic = deterministic
        self._enable_maritime_trade = enable_maritime_trade
        self._features_cache: dict[int, list[str]] = {}
        self._numeric_features_cache: dict[int, list[str]] = {}
        self._rng = random.Random(seed)

    def _build_obs(self, game, color: Optional[Color] = None):
        """Rebuild the gym observation from ``color``'s perspective (default self)."""
        color = color if color is not None else self.color
        num_players = len(game.state.players)
        if num_players not in self._features_cache:
            self._features_cache[num_players] = get_feature_ordering(
                num_players, self._map_type
            )
            self._numeric_features_cache[num_players] = [
                f for f in self._features_cache[num_players] if not is_graph_feature(f)
            ]
        features = self._features_cache[num_players]

        sample = create_sample(game, color)
        if self._representation == "vector":
            return np.array([sample[f] for f in features], dtype=np.float32)
        numeric_features = self._numeric_features_cache[num_players]
        board = create_board_tensor(game, color, channels_first=True).astype(
            np.float32
        )
        numeric = np.array([sample[f] for f in numeric_features], dtype=np.float32)
        return {"board": board, "numeric": numeric}

    def _build_mask(self, player_colors, playable_actions):
        """Boolean action mask matching training (maritime stripped if disabled)."""
        action_array = get_action_array(player_colors, self._map_type)
        valid_idx = {
            to_action_space(a, player_colors, self._map_type) for a in playable_actions
        }
        mask = np.zeros(len(action_array), dtype=bool)
        for i in valid_idx:
            mask[i] = True
        if not self._enable_maritime_trade:
            for i, (atype, _) in enumerate(action_array):
                if atype in TRADE_ACTION_TYPES:
                    mask[i] = False
        return mask

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        player_colors = tuple(p.color for p in game.state.players)
        obs = self._build_obs(game)
        mask = self._build_mask(player_colors, playable_actions)

        if not mask.any():
            return self._rng.choice(list(playable_actions))

        action_int, _ = self._model.predict(
            obs, action_masks=mask, deterministic=self._deterministic
        )
        action_int = int(action_int)

        decoded = from_action_space(action_int, self.color, player_colors, self._map_type)
        if decoded in playable_actions:
            return decoded
        for a in playable_actions:
            if a.action_type == decoded.action_type and a.value == decoded.value:
                return a
        return self._rng.choice(list(playable_actions))
