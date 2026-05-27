"""Wrapper that removes trade actions from the Catanatron gym env's action mask.

Trading is intentionally out of scope for this project (see README). The
underlying Catanatron action array only exposes `MARITIME_TRADE` (player-to-
player trades aren't part of the gym action space at all), so this wrapper
strips those indices wherever the env surfaces them.
"""
from __future__ import annotations

import gymnasium as gym

from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType

TRADE_ACTION_TYPES: frozenset[ActionType] = frozenset(
    at for at in ActionType if "TRADE" in at.name
)


def trade_action_indices(env: gym.Env) -> set[int]:
    """All action-space indices that correspond to a trade action."""
    u = env.unwrapped
    action_array = get_action_array(u.player_colors, u.map_type)
    return {i for i, (atype, _) in enumerate(action_array) if atype in TRADE_ACTION_TYPES}


class NoTradeWrapper(gym.Wrapper):
    """Strips trade indices from `info["valid_actions"]` and `action_masks()`.

    Notes
    -----
    - Catanatron's bot players never propose trades, so we don't need to
      intercept their decisions; only the agent's action space needs masking.
    - `MaskablePPO` reads masks via either `info["action_mask"]` (if used with
      `ActionMasker`) or by calling `env.action_masks()` directly on
      `SubprocVecEnv`. We override both.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._trade_idx = trade_action_indices(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["valid_actions"] = self._filter_valid(info.get("valid_actions", []))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["valid_actions"] = self._filter_valid(info.get("valid_actions", []))
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> list[bool]:
        mask = self.env.unwrapped.action_masks()
        for i in self._trade_idx:
            mask[i] = False
        return mask

    def get_valid_actions(self) -> list[int]:
        return self._filter_valid(self.env.unwrapped.get_valid_actions())

    def _filter_valid(self, valid: list[int]) -> list[int]:
        return [a for a in valid if a not in self._trade_idx]
