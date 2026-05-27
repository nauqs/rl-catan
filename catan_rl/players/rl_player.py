"""Catanatron Player that wraps a trained MaskablePPO checkpoint."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from catanatron import Player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.action_space import (
    from_action_space,
    get_action_array,
    to_action_space,
)
from catanatron.models.enums import ActionType
from catanatron.models.player import Color

from catan_rl.env.no_trade import TRADE_ACTION_TYPES


class RLPlayer(Player):
    """Acts in a Catanatron game using a MaskablePPO policy.

    Reconstructs the same vector observation the gym env produces, builds an
    action mask from `playable_actions` (with trade indices forced off to match
    training), runs `model.predict(deterministic=True)`, and decodes the chosen
    index back to a Catanatron `Action`.
    """

    def __init__(
        self,
        color: Color,
        model_path: str | Path,
        *,
        map_type: str = "BASE",
        deterministic: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(color)
        self._model = MaskablePPO.load(str(model_path), device="cpu")
        self._map_type = map_type
        self._deterministic = deterministic
        self._features_cache: dict[int, list[str]] = {}
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
        features = self._features_cache[num_players]

        sample = create_sample(game, self.color)
        obs = np.array([sample[f] for f in features], dtype=np.float32)

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
            # All legal actions are trades (rare; shouldn't happen with masking
            # in training). Fall back to picking a random legal action.
            return self._rng.choice(list(playable_actions))

        action_int, _ = self._model.predict(
            obs, action_masks=mask, deterministic=self._deterministic
        )
        action_int = int(action_int)

        decoded = from_action_space(action_int, self.color, player_colors, self._map_type)
        if decoded in playable_actions:
            return decoded
        # Mask should prevent this, but stay safe: pick any matching legal action.
        for a in playable_actions:
            if a.action_type == decoded.action_type and a.value == decoded.value:
                return a
        return self._rng.choice(list(playable_actions))
