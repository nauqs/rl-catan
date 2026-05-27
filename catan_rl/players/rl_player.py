"""Catanatron Player that wraps a trained MaskablePPO checkpoint."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from sb3_contrib import MaskablePPO

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
    ):
        super().__init__(color)
        self._model = MaskablePPO.load(str(model_path), device=device)
        self._map_type = map_type
        self._representation = representation
        self._deterministic = deterministic
        self._features_cache: dict[int, list[str]] = {}
        self._numeric_features_cache: dict[int, list[str]] = {}
        self._rng = random.Random(seed)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        player_colors = tuple(p.color for p in game.state.players)
        action_array = get_action_array(player_colors, self._map_type)
        num_players = len(game.state.players)

        if num_players not in self._features_cache:
            self._features_cache[num_players] = get_feature_ordering(
                num_players, self._map_type
            )
            self._numeric_features_cache[num_players] = [
                f for f in self._features_cache[num_players] if not is_graph_feature(f)
            ]
        features = self._features_cache[num_players]

        sample = create_sample(game, self.color)
        if self._representation == "vector":
            obs = np.array([sample[f] for f in features], dtype=np.float32)
        else:
            numeric_features = self._numeric_features_cache[num_players]
            board = create_board_tensor(game, self.color, channels_first=True).astype(
                np.float32
            )
            numeric = np.array(
                [sample[f] for f in numeric_features], dtype=np.float32
            )
            obs = {"board": board, "numeric": numeric}

        valid_idx = {
            to_action_space(a, player_colors, self._map_type) for a in playable_actions
        }
        mask = np.zeros(len(action_array), dtype=bool)
        for i in valid_idx:
            mask[i] = True
        for i, (atype, _) in enumerate(action_array):
            if atype in TRADE_ACTION_TYPES:
                mask[i] = False

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
