"""Behavioral analysis of a trained RLPlayer: what does it actually *do* in games?

Plays N games as BLUE vs a chosen opponent and aggregates:
  - Initial settlement placements: dice-pip sum, resource variety, port
  - Final building counts: settlements, cities, total VPs
  - Roads built: total, ratio to settlements (proxy for "wasted road" frequency)
  - Knight usage: dev cards bought, knights played, robber-steals against opponent
  - Port usage: settlements/cities on port nodes
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron import Color, Game  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.state_functions import (  # noqa: E402
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_played_dev_cards,
)

from catan_rl.env.shaped_reward import ShapedRewardWrapper  # noqa: E402
from catan_rl.eval.opponents import build_opponent  # noqa: E402
from catan_rl.players.rl_player import RLPlayer  # noqa: E402

_DEV_TYPES = ["KNIGHT", "YEAR_OF_PLENTY", "ROAD_BUILDING", "MONOPOLY", "VICTORY_POINT"]
_DEV_PLAY_TYPES_NAMES = {
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_ROAD_BUILDING",
    "PLAY_MONOPOLY",
}

# Pips = number of ways to roll that value on 2d6 (0 for desert / 7).
_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1, None: 0}


def _node_summary(node_id: int, board) -> dict:
    """For a settlement/city node: pip sum, resources touched, port (if any)."""
    tiles = board.map.adjacent_tiles.get(node_id, [])
    pip_sum = sum(_PIPS.get(t.number, 0) for t in tiles)
    resources = sorted({t.resource for t in tiles if t.resource is not None})
    port = None
    for res, nodes in board.map.port_nodes.items():
        if node_id in nodes:
            port = res if res is not None else "ANY"  # None key = 3:1 generic
            break
    return {
        "node": node_id,
        "pip_sum": pip_sum,
        "resources": resources,
        "n_resources": len(resources),
        "port": port,
    }


def _initial_settlement_nodes(action_records, color: Color) -> list[int]:
    """The first two BUILD_SETTLEMENT actions taken by `color` (initial placement phase)."""
    nodes = []
    for rec in action_records:
        a = rec.action
        if a.color == color and a.action_type == ActionType.BUILD_SETTLEMENT:
            nodes.append(a.value)
            if len(nodes) == 2:
                break
    return nodes


def _analyze_game(game: Game, agent_color: Color) -> dict:
    state = game.state
    board = state.board
    recs = state.action_records

    blue_buildings = [(nid, kind) for nid, (c, kind) in board.buildings.items() if c == agent_color]
    n_settlements = sum(1 for _, k in blue_buildings if k == "SETTLEMENT")
    n_cities = sum(1 for _, k in blue_buildings if k == "CITY")
    blue_nodes = [nid for nid, _ in blue_buildings]

    # Roads — board.roads stores each edge twice ((a,b) and (b,a)). Halve.
    n_roads = sum(1 for color in board.roads.values() if color == agent_color) // 2

    # Initial placement quality
    initial_nodes = _initial_settlement_nodes(recs, agent_color)
    initial_summaries = [_node_summary(n, board) for n in initial_nodes]

    # All final buildings
    all_summaries = [_node_summary(n, board) for n in blue_nodes]

    # Action counts (BLUE only)
    action_counts: Counter = Counter()
    knight_plays = 0
    robber_moves_steal = 0  # MOVE_ROBBER where the action targeted a non-self color
    for rec in recs:
        a = rec.action
        if a.color != agent_color:
            continue
        action_counts[a.action_type.name] += 1
        if a.action_type == ActionType.PLAY_KNIGHT_CARD:
            knight_plays += 1
        if a.action_type == ActionType.MOVE_ROBBER:
            victim = a.value[1] if (a.value and len(a.value) > 1) else None
            if victim is not None and victim != agent_color:
                robber_moves_steal += 1

    # Port usage
    port_settlements = sum(1 for s in all_summaries if s["port"] is not None)

    # ---------- VP source decomposition ----------
    n_set_bldgs = sum(1 for _, k in blue_buildings if k == "SETTLEMENT")
    n_cty_bldgs = sum(1 for _, k in blue_buildings if k == "CITY")
    vp_from_buildings = n_set_bldgs * 1 + n_cty_bldgs * 2
    largest_army = get_largest_army(state)
    has_largest_army = largest_army[0] == agent_color
    longest_color = get_longest_road_color(state)
    has_longest_road = longest_color == agent_color
    vp_cards_in_hand = get_dev_cards_in_hand(state, agent_color, "VICTORY_POINT")
    vp_breakdown = {
        "buildings": vp_from_buildings,
        "longest_road": 2 if has_longest_road else 0,
        "largest_army": 2 if has_largest_army else 0,
        "vp_dev_cards": vp_cards_in_hand,
        "actual_total": get_actual_victory_points(state, agent_color),
    }

    # ---------- Dev-card-type distribution (what they bought) ----------
    dev_drawn: Counter = Counter()
    for rec in recs:
        if (rec.action.color == agent_color
                and rec.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
                and isinstance(rec.result, str)):
            dev_drawn[rec.result] += 1

    # ---------- Reward attribution under Charlesworth shaping ----------
    # All dense terms; uses telescoping for VP_delta so we don't need step-by-step.
    final_vp = get_actual_victory_points(state, agent_color)
    n_dev_plays = sum(get_played_dev_cards(state, agent_color, t)
                      for t in ("KNIGHT", "YEAR_OF_PLENTY", "ROAD_BUILDING", "MONOPOLY"))
    n_move_robber = action_counts.get("MOVE_ROBBER", 0)
    n_discard = action_counts.get("DISCARD_RESOURCE", 0)
    n_city_actions = action_counts.get("BUILD_CITY", 0)
    rewards = {
        "vp_delta": 5.0 * final_vp,                           # telescopes
        "dev_play": ShapedRewardWrapper.DEV_PLAY_REWARD * n_dev_plays,
        "robber": ShapedRewardWrapper.ROBBER_REWARD * n_move_robber,
        "discard": ShapedRewardWrapper.DISCARD_PENALTY * n_discard,
        "city_bonus": ShapedRewardWrapper.CITY_BONUS * n_city_actions,
        "terminal_win": ShapedRewardWrapper.WIN_REWARD if game.winning_color() == agent_color else 0.0,
    }
    rewards["dense_total"] = sum(rewards[k] for k in ("vp_delta", "dev_play", "robber", "discard", "city_bonus"))
    rewards["total"] = rewards["dense_total"] + rewards["terminal_win"]

    return {
        "winner": game.winning_color(),
        "won": game.winning_color() == agent_color,
        "vp": get_actual_victory_points(state, agent_color),
        "turns": state.num_turns,
        "n_settlements": n_settlements,
        "n_cities": n_cities,
        "n_roads": n_roads,
        "knight_plays": knight_plays,
        "robber_moves_to_opponent": robber_moves_steal,
        "buy_dev_card": action_counts.get("BUY_DEVELOPMENT_CARD", 0),
        "build_settlement_actions": action_counts.get("BUILD_SETTLEMENT", 0),
        "build_city_actions": action_counts.get("BUILD_CITY", 0),
        "build_road_actions": action_counts.get("BUILD_ROAD", 0),
        "initial_summaries": initial_summaries,
        "all_summaries": all_summaries,
        "port_settlement_count": port_settlements,
        "vp_breakdown": vp_breakdown,
        "dev_drawn": dict(dev_drawn),
        "rewards": rewards,
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n if n else 0.0

    initial_flat = [s for r in rows for s in r["initial_summaries"]]
    final_flat = [s for r in rows for s in r["all_summaries"]]

    def pip_stats(summaries: list[dict]) -> dict:
        if not summaries:
            return {"mean_pip_sum": 0.0, "mean_n_resources": 0.0, "frac_3plus_resources": 0.0,
                    "frac_on_port": 0.0}
        pip = sum(s["pip_sum"] for s in summaries) / len(summaries)
        nres = sum(s["n_resources"] for s in summaries) / len(summaries)
        frac3 = sum(1 for s in summaries if s["n_resources"] >= 3) / len(summaries)
        port = sum(1 for s in summaries if s["port"] is not None) / len(summaries)
        return {"mean_pip_sum": pip, "mean_n_resources": nres,
                "frac_3plus_resources": frac3, "frac_on_port": port}

    # Per-resource counts at initial placement
    res_counts: Counter = Counter()
    for s in initial_flat:
        for r in s["resources"]:
            res_counts[r] += 1
    initial_resource_dist = {r: c / len(initial_flat) for r, c in res_counts.items()} if initial_flat else {}

    # ---------- VP source aggregate ----------
    vp_keys = ("buildings", "longest_road", "largest_army", "vp_dev_cards")
    vp_avg = {k: sum(r["vp_breakdown"][k] for r in rows) / n for k in vp_keys} if n else {}
    vp_share = {k: (vp_avg[k] / mean("vp")) if mean("vp") else 0.0 for k in vp_keys}

    # ---------- Dev card type aggregate ----------
    dev_total: Counter = Counter()
    for r in rows:
        for k, v in r["dev_drawn"].items():
            dev_total[k] += v
    dev_per_game = {k: dev_total.get(k, 0) / n for k in _DEV_TYPES} if n else {}

    # ---------- Reward attribution aggregate ----------
    rew_keys = ("vp_delta", "dev_play", "robber", "discard", "city_bonus", "terminal_win")
    rew_avg = {k: sum(r["rewards"][k] for r in rows) / n for k in rew_keys} if n else {}
    total_avg = sum(abs(v) for v in rew_avg.values()) or 1.0
    rew_share_abs = {k: abs(rew_avg[k]) / total_avg for k in rew_keys}

    # ---------- City distribution ----------
    city_hist = Counter(r["n_cities"] for r in rows)
    city_hist_norm = {str(k): v / n for k, v in sorted(city_hist.items())} if n else {}

    # ---------- Win vs loss conditional ----------
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]

    def win_loss_avg(group: list[dict], key: str) -> float:
        return sum(g[key] for g in group) / len(group) if group else 0.0

    win_loss = {
        "n_wins": len(wins),
        "n_losses": len(losses),
        "wins_mean_turns": win_loss_avg(wins, "turns"),
        "losses_mean_turns": win_loss_avg(losses, "turns"),
        "wins_mean_cities": win_loss_avg(wins, "n_cities"),
        "losses_mean_cities": win_loss_avg(losses, "n_cities"),
        "wins_mean_vp": win_loss_avg(wins, "vp"),
        "losses_mean_vp": win_loss_avg(losses, "vp"),
    }

    return {
        "n_games": n,
        "winrate": mean("won"),
        "mean_vp": mean("vp"),
        "mean_turns": mean("turns"),
        "mean_settlements_at_end": mean("n_settlements"),
        "mean_cities_at_end": mean("n_cities"),
        "mean_roads_at_end": mean("n_roads"),
        "roads_per_settlement_total": (
            sum(r["n_roads"] for r in rows) / max(1, sum(r["n_settlements"] + r["n_cities"] for r in rows))
        ),
        "mean_city_upgrade_actions": mean("build_city_actions"),
        "mean_settlement_build_actions": mean("build_settlement_actions"),
        "mean_road_build_actions": mean("build_road_actions"),
        "mean_dev_cards_bought": mean("buy_dev_card"),
        "mean_knights_played": mean("knight_plays"),
        "mean_robber_steals_from_opponent": mean("robber_moves_to_opponent"),
        "initial_placement": pip_stats(initial_flat),
        "initial_resource_frequency": initial_resource_dist,
        "final_buildings_avg": pip_stats(final_flat),
        "port_usage_per_game": sum(r["port_settlement_count"] for r in rows) / n if n else 0.0,
        "vp_source_avg": vp_avg,
        "vp_source_share": vp_share,
        "dev_draws_per_game": dev_per_game,
        "reward_attribution_avg": rew_avg,
        "reward_attribution_share_abs": rew_share_abs,
        "cities_per_game_distribution": city_hist_norm,
        "win_vs_loss": win_loss,
    }


def _fmt_dict(d: dict, *, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(_fmt_dict(v, indent=indent + 1))
        elif isinstance(v, float):
            out.append(f"{pad}{k}: {v:.3f}")
        else:
            out.append(f"{pad}{k}: {v}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyze")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="weighted",
                   choices=["random", "weighted", "value", "alphabeta1", "alphabeta2"])
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--representation", default="mixed", choices=["mixed", "vector"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for g in range(args.games):
        agent = RLPlayer(Color.BLUE, args.checkpoint, representation=args.representation)
        opp = build_opponent(args.opponent, Color.RED)
        game = Game([agent, opp], seed=args.seed + g)
        game.play()
        rows.append(_analyze_game(game, Color.BLUE))
        if (g + 1) % 25 == 0:
            print(f"  ...{g+1}/{args.games} ({time.perf_counter()-t0:.1f}s)", flush=True)

    summary = _aggregate(rows)
    summary["opponent"] = args.opponent
    summary["checkpoint"] = str(args.checkpoint)
    summary["elapsed_seconds"] = time.perf_counter() - t0

    print("\n=== BEHAVIORAL ANALYSIS ===")
    print(_fmt_dict(summary))

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
