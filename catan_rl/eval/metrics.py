"""Cheap per-game behavioral metrics collected during training eval callbacks.

All metrics are derived from the final game state + action_records in O(n)
time (n = game length, typically 150-300). No extra simulation needed.
"""
from __future__ import annotations

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points

_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
_DEV_PLAY_TYPES = frozenset({
    ActionType.PLAY_KNIGHT_CARD,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_MONOPOLY,
})


def collect(game: Game, agent_color: Color) -> dict:
    """Extract behavioral metrics from a finished game. Returns a flat dict of scalars."""
    state = game.state
    board = state.board
    records = state.action_records

    # --- buildings ---
    blue_bldgs = [(nid, kind) for nid, (c, kind) in board.buildings.items() if c == agent_color]
    n_settlements = sum(1 for _, k in blue_bldgs if k == "SETTLEMENT")
    n_cities = sum(1 for _, k in blue_bldgs if k == "CITY")

    # --- roads ---
    n_roads = sum(1 for c in board.roads.values() if c == agent_color) // 2
    road_length = board.road_lengths.get(agent_color, 0)

    # --- initial placement quality (first 2 BUILD_SETTLEMENT by agent) ---
    init_pip, init_res = [], []
    for rec in records:
        a = rec.action
        if a.color != agent_color or a.action_type != ActionType.BUILD_SETTLEMENT:
            continue
        tiles = board.map.adjacent_tiles.get(a.value, [])
        init_pip.append(sum(_PIPS.get(t.number, 0) for t in tiles))
        init_res.append(len({t.resource for t in tiles if t.resource is not None}))
        if len(init_pip) == 2:
            break
    mean_init_pip = sum(init_pip) / len(init_pip) if init_pip else 0.0
    mean_init_res = sum(init_res) / len(init_res) if init_res else 0.0

    # --- action counters (agent only) ---
    n_dev_purchased = 0
    n_dev_played = 0          # non-VP dev cards played
    n_knights_opponent = 0    # robber targeting an opponent node
    n_knights_self_desert = 0 # robber to desert or own tile (wasted)
    n_discard = 0
    n_city_built = 0

    for rec in records:
        a = rec.action
        if a.color != agent_color:
            continue
        t = a.action_type
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            n_dev_purchased += 1
        elif t in _DEV_PLAY_TYPES:
            n_dev_played += 1
        elif t == ActionType.MOVE_ROBBER:
            victim = a.value[1] if (a.value and len(a.value) > 1) else None
            if victim is not None and victim != agent_color:
                n_knights_opponent += 1
            else:
                n_knights_self_desert += 1
        elif t == ActionType.DISCARD_RESOURCE:
            n_discard += 1
        elif t == ActionType.BUILD_CITY:
            n_city_built += 1

    won = game.winning_color() == agent_color
    vp = get_actual_victory_points(state, agent_color)

    return {
        "won": float(won),
        "vp": vp,
        "n_settlements": n_settlements,
        "n_cities": n_cities,
        "n_roads": n_roads,
        "road_length": road_length,
        "mean_init_pip": mean_init_pip,
        "mean_init_res": mean_init_res,
        "n_dev_purchased": n_dev_purchased,
        "n_dev_played": n_dev_played,
        "n_dev_purchased_no_vp": n_dev_purchased - sum(
            1 for rec in records
            if rec.action.color == agent_color
            and rec.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
            and isinstance(rec.result, str) and rec.result == "VICTORY_POINT"
        ),
        "n_knights_opponent": n_knights_opponent,
        "n_knights_self_desert": n_knights_self_desert,
        "n_discard": n_discard,
        "n_city_built": n_city_built,
        "turns": state.num_turns,
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Average a list of per-game metric dicts."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
