"""Registry of opponent factories used by the ladder eval."""
from __future__ import annotations

from typing import Callable

from catanatron import Player, RandomPlayer
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

OpponentFactory = Callable[[Color], Player]

# Ladder of opponents, roughly weakest -> strongest. `vp` (VictoryPointPlayer)
# is the natural intermediate between weighted and value: picks randomly among
# actions that immediately increase VP, else random.
OPPONENTS: dict[str, OpponentFactory] = {
    "random": lambda c: RandomPlayer(c),
    "weighted": lambda c: WeightedRandomPlayer(c),
    "vp": lambda c: VictoryPointPlayer(c),
    "value": lambda c: ValueFunctionPlayer(c),
    "alphabeta1": lambda c: AlphaBetaPlayer(c, depth=1),
    "alphabeta2": lambda c: AlphaBetaPlayer(c, depth=2),
}


def build_opponent(name: str, color: Color) -> Player:
    if name not in OPPONENTS:
        raise KeyError(f"unknown opponent '{name}'; known: {sorted(OPPONENTS)}")
    return OPPONENTS[name](color)
