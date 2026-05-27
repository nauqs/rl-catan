"""Ladder eval: run a candidate Player against each opponent in 1v1 / 4-player."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from catanatron import Color, Game, Player
from catanatron.state_functions import get_actual_victory_points

from catan_rl.eval.opponents import OPPONENTS, build_opponent

Mode = Literal["1v1", "4p"]
CandidateFactory = Callable[[Color], Player]

# Color slots used per mode. Candidate is always BLUE; opponents take the rest.
_MODE_COLORS: dict[Mode, list[Color]] = {
    "1v1": [Color.RED],
    "4p": [Color.RED, Color.WHITE, Color.ORANGE],
}


@dataclass
class OpponentResult:
    opponent: str
    mode: Mode
    n_games: int
    wins: int
    mean_vp: float
    mean_turns: float
    winrate_ci95: tuple[float, float]
    seconds: float

    @property
    def winrate(self) -> float:
        return self.wins / self.n_games if self.n_games else 0.0

    def to_dict(self) -> dict:
        return {
            "opponent": self.opponent,
            "mode": self.mode,
            "n_games": self.n_games,
            "wins": self.wins,
            "winrate": self.winrate,
            "winrate_ci95_lo": self.winrate_ci95[0],
            "winrate_ci95_hi": self.winrate_ci95[1],
            "mean_vp": self.mean_vp,
            "mean_turns": self.mean_turns,
            "seconds": self.seconds,
        }


@dataclass
class LadderResult:
    candidate: str
    by_opponent: list[OpponentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "results": [r.to_dict() for r in self.by_opponent],
        }


def wilson_ci95(wins: int, n: int) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _play_one(
    candidate_factory: CandidateFactory,
    opponent_name: str,
    mode: Mode,
    seed: int,
) -> tuple[bool, int, int]:
    candidate = candidate_factory(Color.BLUE)
    opp_colors = _MODE_COLORS[mode]
    opponents = [build_opponent(opponent_name, c) for c in opp_colors]
    game = Game([candidate] + opponents, seed=seed)
    game.play()
    won = game.winning_color() == Color.BLUE
    vp = get_actual_victory_points(game.state, Color.BLUE)
    return won, vp, game.state.num_turns


def run_ladder(
    candidate_factory: CandidateFactory,
    *,
    candidate_name: str = "candidate",
    opponents: Iterable[str] = tuple(OPPONENTS),
    n_games: int = 20,
    modes: Iterable[Mode] = ("1v1", "4p"),
    base_seed: int = 0,
) -> LadderResult:
    """Play `n_games` of `candidate` against each opponent in each mode."""
    result = LadderResult(candidate=candidate_name)
    for mode in modes:
        for opp in opponents:
            t0 = time.perf_counter()
            wins = 0
            vp_total = 0
            turns_total = 0
            for g in range(n_games):
                seed = base_seed + g + hash((mode, opp)) % 10_000
                won, vp, turns = _play_one(candidate_factory, opp, mode, seed)
                wins += int(won)
                vp_total += vp
                turns_total += turns
            elapsed = time.perf_counter() - t0
            result.by_opponent.append(
                OpponentResult(
                    opponent=opp,
                    mode=mode,
                    n_games=n_games,
                    wins=wins,
                    mean_vp=vp_total / n_games,
                    mean_turns=turns_total / n_games,
                    winrate_ci95=wilson_ci95(wins, n_games),
                    seconds=elapsed,
                )
            )
    return result
