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
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_length,
)

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
        self._prev_longest: int = 0
        self._init_res_union: set = set()
        self._init_settle_count: int = 0

    # ---- public hook for AnnealingCallback ------------------------------------
    def set_dense_factor(self, factor: float) -> None:
        self.dense_factor = float(factor)

    # ---- gym.Wrapper API ------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        self._prev_vp = get_actual_victory_points(u.game.state, u.p0.color)
        self._prev_records_len = len(u.game.state.action_records)
        self._prev_longest = get_longest_road_length(u.game.state, u.p0.color)
        self._init_res_union = set()
        self._init_settle_count = 0
        return obs, info

    @staticmethod
    def _tile_adjacency(board, tile_coord, agent_color):
        """(tile_pip, opp_adjacent, self_adjacent) for a robber target tile."""
        tile = board.map.land_tiles.get(tile_coord)
        if tile is None:
            return 0, False, False
        pip = _PIPS.get(tile.number, 0)
        opp_adj = self_adj = False
        for node_id in tile.nodes.values():
            owner = board.buildings.get(node_id)
            if owner is None:
                continue
            color = owner[0]
            if color == agent_color:
                self_adj = True
            else:
                opp_adj = True
        return pip, opp_adj, self_adj

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if not self.enabled or not self.cfg.enabled:
            return obs, base_reward, terminated, truncated, info

        u = self.env.unwrapped
        state = u.game.state
        board = state.board
        agent_color = u.p0.color
        cfg = self.cfg

        annealed = 0.0      # VP/building/placement signals (× dense_factor)
        persistent = 0.0    # efficiency/tactics (always on): roads/maritime/robber

        # VP delta (annealed)
        cur_vp = get_actual_victory_points(state, agent_color)
        annealed += cfg.vp_coef * (cur_vp - self._prev_vp)
        self._prev_vp = cur_vp

        # Per-step penalty (annealed)
        annealed += cfg.per_turn_penalty

        # Action-record-driven terms (BLUE's actions only)
        records = state.action_records
        for rec in records[self._prev_records_len :]:
            act = rec.action
            if act.color != agent_color:
                continue
            atype = act.action_type

            if atype in _DEV_PLAY_TYPES:
                annealed += cfg.dev_play_reward

            elif atype == ActionType.MOVE_ROBBER:
                # value = (tile_coord, victim_color). PERSISTENT: +pip×opp, -pip×self, +steal.
                target_tile_coord = act.value[0] if act.value else None
                victim = act.value[1] if (act.value and len(act.value) > 1) else None
                if target_tile_coord is not None:
                    pip, opp_adj, self_adj = self._tile_adjacency(
                        board, target_tile_coord, agent_color
                    )
                    if opp_adj:
                        persistent += cfg.robber_block_opp_pip * pip
                    if self_adj:
                        persistent += cfg.robber_block_self_pip * pip  # coef is negative
                # legacy (annealed) terms, kept for back-compat
                opp_victim = victim is not None and victim != agent_color
                if opp_victim:
                    annealed += cfg.robber_target_opponent
                    persistent += cfg.robber_steal
                if cfg.robber_blocks_high_pip and opp_victim and target_tile_coord is not None:
                    tile = board.map.land_tiles.get(target_tile_coord)
                    if tile is not None:
                        annealed += cfg.robber_blocks_high_pip * _PIPS.get(tile.number, 0)

            elif atype == ActionType.MARITIME_TRADE:
                # value = (give1..give4, receive); n_gives determines the rate.
                n_gives = sum(1 for x in act.value[:-1] if x is not None)
                persistent += cfg.maritime_penalty * max(0, n_gives - 1)

            elif atype == ActionType.BUILD_ROAD:
                persistent += cfg.road_cost
                cur_longest = get_longest_road_length(state, agent_color)
                if cur_longest > self._prev_longest:
                    persistent += cfg.longest_road_delta * (cur_longest - self._prev_longest)
                self._prev_longest = cur_longest

            elif atype == ActionType.DISCARD_RESOURCE:
                annealed += cfg.discard_penalty

            elif atype == ActionType.BUILD_CITY:
                annealed += cfg.city_bonus

            elif atype == ActionType.BUILD_SETTLEMENT:
                node_id = act.value
                tiles = board.map.adjacent_tiles.get(node_id, [])
                resources = {t.resource for t in tiles if t.resource is not None}
                # Initial-placement (first 2 settlements): pip per-settlement + combined coverage.
                if self._init_settle_count < 2:
                    pip_sum = sum(_PIPS.get(t.number, 0) for t in tiles)
                    annealed += cfg.initial_placement_pip * pip_sum
                    annealed += cfg.initial_placement_diversity * len(resources)
                    self._init_res_union |= resources
                    self._init_settle_count += 1
                    if self._init_settle_count == 2 and cfg.placement_coverage:
                        annealed += cfg.placement_coverage.get(len(self._init_res_union), 0.0)
                else:
                    annealed += cfg.settlement_bonus

        self._prev_records_len = len(records)

        # Terminal (not annealed). Treat truncation (TURNS_LIMIT) as a loss
        # so the agent is actively penalized for stalling.
        terminal = 0.0
        if terminated or truncated:
            winner = u.game.winning_color()
            terminal = cfg.win_reward if winner == agent_color else cfg.loss_reward

        return (
            obs,
            terminal + self.dense_factor * annealed + persistent,
            terminated, truncated, info,
        )
