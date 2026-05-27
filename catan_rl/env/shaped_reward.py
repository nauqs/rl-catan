"""Charlesworth-style reward shaping for the trainable BLUE player.

Values lifted from https://github.com/henrycharlesworth/settlers_of_catan_RL
(`env/wrapper.py`). Dense terms are scaled by `dense_factor` so an external
`AnnealingCallback` can taper them linearly toward 0 over training; terminal
+500 is never scaled.
"""
from __future__ import annotations

import gymnasium as gym

from catanatron.models.enums import ActionType
from catanatron.state_functions import get_actual_victory_points


_DEV_PLAY_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_MONOPOLY,
    }
)


class ShapedRewardWrapper(gym.Wrapper):
    """Replaces Catanatron's ±1 terminal reward with Charlesworth's dense + sparse scheme.

    Applies only to the trainable agent (`env.unwrapped.p0`, BLUE by default).
    Opponent actions are ignored when scanning `state.action_records`.
    """

    WIN_REWARD = 500.0
    VP_COEF = 5.0
    DEV_PLAY_REWARD = 5.0
    ROBBER_REWARD = 1.0
    DISCARD_PENALTY = -0.3
    CITY_BONUS = 2.5

    def __init__(self, env: gym.Env, *, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        self.dense_factor: float = 1.0
        self._prev_vp: int = 0
        self._prev_records_len: int = 0

    # ---- public hook for AnnealingCallback ------------------------------------
    def set_dense_factor(self, factor: float) -> None:
        self.dense_factor = float(factor)

    # ---- gym.Wrapper API ------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        self._prev_vp = get_actual_victory_points(u.game.state, u.p0.color)
        self._prev_records_len = len(u.game.state.action_records)
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if not self.enabled:
            return obs, base_reward, terminated, truncated, info

        u = self.env.unwrapped

        dense = 0.0
        cur_vp = get_actual_victory_points(u.game.state, u.p0.color)
        dense += self.VP_COEF * (cur_vp - self._prev_vp)
        self._prev_vp = cur_vp

        records = u.game.state.action_records
        for rec in records[self._prev_records_len :]:
            if rec.action.color != u.p0.color:
                continue
            atype = rec.action.action_type
            if atype in _DEV_PLAY_TYPES:
                dense += self.DEV_PLAY_REWARD
            elif atype == ActionType.MOVE_ROBBER:
                dense += self.ROBBER_REWARD
            elif atype == ActionType.DISCARD_RESOURCE:
                dense += self.DISCARD_PENALTY
            elif atype == ActionType.BUILD_CITY:
                dense += self.CITY_BONUS
        self._prev_records_len = len(records)

        terminal = 0.0
        if terminated and u.game.winning_color() == u.p0.color:
            terminal = self.WIN_REWARD

        return obs, terminal + self.dense_factor * dense, terminated, truncated, info
