"""Smoke test the ladder harness over all 5 opponents (1v1, tiny game counts)."""
from __future__ import annotations

from catanatron import RandomPlayer

from catan_rl.eval.ladder import run_ladder
from catan_rl.eval.opponents import OPPONENTS


def test_ladder_smoke_random_candidate() -> None:
    result = run_ladder(
        lambda color: RandomPlayer(color),
        candidate_name="random",
        opponents=list(OPPONENTS),
        n_games=2,
        modes=("1v1",),
    )
    assert {r.opponent for r in result.by_opponent} == set(OPPONENTS)
    for r in result.by_opponent:
        assert r.n_games == 2
        assert r.mode == "1v1"
        assert 0.0 <= r.winrate <= 1.0
        lo, hi = r.winrate_ci95
        assert 0.0 <= lo <= r.winrate <= hi <= 1.0
        assert r.mean_turns > 0
