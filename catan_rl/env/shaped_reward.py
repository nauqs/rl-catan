"""Reward shaping for the trainable BLUE player.

Charlesworth-style base (from `settlers_of_catan_RL/env/wrapper.py`) plus
Phase-2a refinements. All coefficients live in `RewardConfig`; setting any
term to 0 disables it. Dense terms are scaled by `dense_factor` so the
`AnnealingCallback` can taper them toward 0 over training. Terminal
win/loss rewards are never scaled.
"""
from __future__ import annotations

import gymnasium as gym

from catanatron.models.enums import ActionType
from catanatron.state_functions import get_actual_victory_points

from catan_rl.utils.config import RewardConfig


_DEV_PLAY_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_MONOPOLY,
    }
)

_PIPS: dict[int | None, int] = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    7: 0,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
    None: 0,  # desert
}


class ShapedRewardWrapper(gym.Wrapper):
    """Reward shaping wrapper. Reads all coefficients from a `RewardConfig`.

    Back-compat: legacy class-constants (`WIN_REWARD`, `VP_COEF`, ...) are
    preserved so existing tests / analysis scripts can still reference them.
    """

    # Legacy class-constants (Phase-1 defaults, for back-compat references).
    WIN_REWARD = 500.0
    VP_COEF = 5.0
    DEV_PLAY_REWARD = 5.0
    ROBBER_REWARD = 1.0
    DISCARD_PENALTY = -0.3
    CITY_BONUS = 2.5

    def __init__(
        self,
        env: gym.Env,
        *,
        enabled: bool = True,
        config: RewardConfig | None = None,
    ):
        super().__init__(env)
        self.enabled = enabled
        self.cfg = config or RewardConfig()
        # If caller passed enabled=False, mirror it on the config view too.
        if not enabled:
            self.cfg = self.cfg.model_copy(update={"enabled": False})
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
        if not self.enabled or not self.cfg.enabled:
            return obs, base_reward, terminated, truncated, info

        u = self.env.unwrapped
        state = u.game.state
        board = state.board
        agent_color = u.p0.color
        cfg = self.cfg

        dense = 0.0

        # VP delta
        cur_vp = get_actual_victory_points(state, agent_color)
        dense += cfg.vp_coef * (cur_vp - self._prev_vp)
        self._prev_vp = cur_vp

        # Per-step penalty (always applied to every env step)
        dense += cfg.per_turn_penalty

        # Action-record-driven dense terms (BLUE's actions only)
        records = state.action_records
        for rec in records[self._prev_records_len :]:
            act = rec.action
            if act.color != agent_color:
                continue
            atype = act.action_type

            if atype in _DEV_PLAY_TYPES:
                dense += cfg.dev_play_reward

            elif atype == ActionType.MOVE_ROBBER:
                # value = (tile_coord, victim_color)
                target_tile_coord = act.value[0] if act.value else None
                victim = act.value[1] if (act.value and len(act.value) > 1) else None
                opp_adjacent = victim is not None and victim != agent_color
                if opp_adjacent:
                    dense += cfg.robber_target_opponent
                if cfg.robber_blocks_high_pip and opp_adjacent and target_tile_coord is not None:
                    tile = board.map.land_tiles.get(target_tile_coord)
                    if tile is not None:
                        dense += cfg.robber_blocks_high_pip * _PIPS.get(tile.number, 0)

            elif atype == ActionType.DISCARD_RESOURCE:
                dense += cfg.discard_penalty

            elif atype == ActionType.BUILD_CITY:
                dense += cfg.city_bonus

            elif atype == ActionType.BUILD_SETTLEMENT:
                # Initial-placement bonuses fire only during the initial-build phase.
                # We must check whether the settlement was placed during initial phase.
                # Heuristic: the agent has at most 2 initial settlements; count agent
                # settlements before this record was applied to decide.
                if cfg.initial_placement_pip != 0.0 or cfg.initial_placement_diversity != 0.0:
                    node_id = act.value
                    tiles = board.map.adjacent_tiles.get(node_id, [])
                    pip_sum = sum(_PIPS.get(t.number, 0) for t in tiles)
                    resources = {t.resource for t in tiles if t.resource is not None}
                    # We only count first two settlement-builds (initial phase).
                    # Counting from action_records is robust to state mutations.
                    settlements_so_far = sum(
                        1 for r2 in records
                        if r2.action.color == agent_color
                        and r2.action.action_type == ActionType.BUILD_SETTLEMENT
                        and records.index(r2) <= records.index(rec)
                    )
                    if settlements_so_far <= 2:
                        dense += cfg.initial_placement_pip * pip_sum
                        dense += cfg.initial_placement_diversity * len(resources)

        self._prev_records_len = len(records)

        # Terminal (not annealed). Treat truncation (TURNS_LIMIT) as a loss
        # so the agent is actively penalized for stalling.
        terminal = 0.0
        if terminated or truncated:
            winner = u.game.winning_color()
            if winner == agent_color:
                terminal = cfg.win_reward
            else:
                terminal = cfg.loss_reward

        return obs, terminal + self.dense_factor * dense, terminated, truncated, info
