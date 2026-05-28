"""OpponentSamplers: how training envs pick the opposing player(s) per episode."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from catanatron import Color, Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


OpponentFactory = Callable[[Color], Player]


_BUILTIN_OPPONENTS: dict[str, OpponentFactory] = {
    "random": lambda c: RandomPlayer(c),
    "weighted": lambda c: WeightedRandomPlayer(c),
    "value": lambda c: ValueFunctionPlayer(c),
    "alphabeta1": lambda c: AlphaBetaPlayer(c, depth=1),
    "alphabeta2": lambda c: AlphaBetaPlayer(c, depth=2),
}


def _builtin_factory(name: str) -> OpponentFactory:
    if name not in _BUILTIN_OPPONENTS:
        raise KeyError(f"unknown opponent {name!r}; known: {sorted(_BUILTIN_OPPONENTS)}")
    return _BUILTIN_OPPONENTS[name]


class OpponentSampler(ABC):
    """Returns a factory `Color -> Player` each time `sample()` is called."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> OpponentFactory: ...


class SingleOpponent(OpponentSampler):
    """Always the same opponent class (Phase-1 behavior)."""

    def __init__(self, name: str):
        self.name = name
        self._factory = _builtin_factory(name)

    def sample(self, rng):
        return self._factory


class MixedOpponent(OpponentSampler):
    """Weighted sample from a fixed bag of opponent classes."""

    def __init__(self, mix: dict[str, float]):
        if not mix:
            raise ValueError("MixedOpponent requires a non-empty mix")
        self.names = list(mix.keys())
        weights = np.array([mix[n] for n in self.names], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(f"invalid weights: {mix}")
        self.probs = weights / weights.sum()
        # Pre-resolve builtin factories; unknown names error early.
        self._factories = [_builtin_factory(n) for n in self.names]

    def sample(self, rng):
        idx = int(rng.choice(len(self.names), p=self.probs))
        return self._factories[idx]


class MixedWithSelf(OpponentSampler):
    """Like MixedOpponent but one slot ('self') loads a periodically-refreshed
    snapshot of the current policy from `snapshot_path`. If the snapshot doesn't
    exist yet, the 'self' slot falls back to `fallback_name`.
    """

    SELF_KEY = "self"

    def __init__(
        self,
        mix: dict[str, float],
        snapshot_path: Path | str,
        *,
        fallback_name: str = "weighted",
        representation: str = "mixed",
    ):
        if self.SELF_KEY not in mix:
            raise ValueError(f"MixedWithSelf requires a '{self.SELF_KEY}' entry in mix")
        self.snapshot_path = Path(snapshot_path)
        self.fallback_name = fallback_name
        self.fallback_factory = _builtin_factory(fallback_name)
        self.representation = representation

        self.names = list(mix.keys())
        weights = np.array([mix[n] for n in self.names], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(f"invalid weights: {mix}")
        self.probs = weights / weights.sum()
        self._builtin_factories: dict[str, OpponentFactory] = {
            n: _builtin_factory(n) for n in self.names if n != self.SELF_KEY
        }

    def sample(self, rng):
        idx = int(rng.choice(len(self.names), p=self.probs))
        name = self.names[idx]
        if name == self.SELF_KEY:
            return self._self_factory()
        return self._builtin_factories[name]

    def _self_factory(self) -> OpponentFactory:
        if not self.snapshot_path.exists():
            return self.fallback_factory
        rep = self.representation
        snapshot_path = self.snapshot_path

        def _factory(color: Color) -> Player:
            # Lazy import to keep module import cheap.
            from catan_rl.players.rl_player import RLPlayer

            return RLPlayer(color, snapshot_path, representation=rep, deterministic=False)

        return _factory


def build_sampler(opponent_cfg, *, snapshot_path: Path | str | None = None,
                  representation: str = "mixed") -> OpponentSampler:
    """Resolve an `OpponentConfig` to a concrete `OpponentSampler`."""
    if opponent_cfg.sampler == "single":
        # Use first key in mix as the single opponent.
        name = next(iter(opponent_cfg.mix))
        return SingleOpponent(name)
    if opponent_cfg.sampler == "mixed":
        return MixedOpponent(dict(opponent_cfg.mix))
    if opponent_cfg.sampler == "mixed_with_self":
        if snapshot_path is None:
            raise ValueError("mixed_with_self sampler requires snapshot_path")
        return MixedWithSelf(
            dict(opponent_cfg.mix),
            snapshot_path=snapshot_path,
            fallback_name=opponent_cfg.self_fallback,
            representation=representation,
        )
    raise ValueError(f"unknown sampler {opponent_cfg.sampler!r}")
