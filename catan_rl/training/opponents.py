"""OpponentSamplers: how training envs pick the opposing player(s) per episode."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Mapping

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

    # Optional hook for schedulers that need to know the current training step.
    # Default no-op so non-scheduled samplers don't have to care.
    def set_step(self, step: int) -> None:
        return None


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
        enable_maritime_trade: bool = False,
    ):
        if self.SELF_KEY not in mix:
            raise ValueError(f"MixedWithSelf requires a '{self.SELF_KEY}' entry in mix")
        self.snapshot_path = Path(snapshot_path)
        self.fallback_name = fallback_name
        self.fallback_factory = _builtin_factory(fallback_name)
        self.representation = representation
        self.enable_maritime_trade = enable_maritime_trade

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
        enable_maritime = self.enable_maritime_trade

        def _factory(color: Color) -> Player:
            # Lazy import to keep module import cheap.
            from catan_rl.players.rl_player import RLPlayer

            return RLPlayer(
                color, snapshot_path, representation=rep, deterministic=False,
                enable_maritime_trade=enable_maritime,
            )

        return _factory


class ScheduledOpponent(OpponentSampler):
    """Linearly interpolates between `start_mix` and `end_mix` over training.

    The current weights are computed from `current_step`:
      - step <= schedule_start_step: pure start_mix
      - step >= schedule_end_step:   pure end_mix
      - otherwise: linear interpolation

    `current_step` is pushed by an external `OpponentScheduleCallback`
    via the env wrapper's `set_opponent_step` -> sampler `set_step`.

    'self' is treated like `MixedWithSelf`: when sampled, loads a frozen
    snapshot from `snapshot_path`; falls back to `fallback_name` if the
    snapshot doesn't exist yet.
    """

    SELF_KEY = "self"

    def __init__(
        self,
        start_mix: Mapping[str, float],
        end_mix: Mapping[str, float],
        *,
        schedule_start_step: int,
        schedule_end_step: int,
        snapshot_path: Path | str | None = None,
        fallback_name: str = "weighted",
        representation: str = "mixed",
        enable_maritime_trade: bool = False,
    ):
        if schedule_end_step <= schedule_start_step:
            raise ValueError("schedule_end_step must be > schedule_start_step")
        self.names = sorted(set(start_mix.keys()) | set(end_mix.keys()))
        self.start_weights = self._normalize(
            np.array([start_mix.get(n, 0.0) for n in self.names], dtype=float)
        )
        self.end_weights = self._normalize(
            np.array([end_mix.get(n, 0.0) for n in self.names], dtype=float)
        )
        self.schedule_start_step = int(schedule_start_step)
        self.schedule_end_step = int(schedule_end_step)
        self.current_step: int = 0

        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self.fallback_name = fallback_name
        self.fallback_factory = _builtin_factory(fallback_name)
        self.representation = representation
        self.enable_maritime_trade = enable_maritime_trade
        self._builtin_factories: dict[str, OpponentFactory] = {
            n: _builtin_factory(n) for n in self.names if n != self.SELF_KEY
        }
        if self.SELF_KEY in self.names and self.snapshot_path is None:
            raise ValueError("ScheduledOpponent with 'self' in mix requires snapshot_path")

    @staticmethod
    def _normalize(w: np.ndarray) -> np.ndarray:
        if w.sum() <= 0:
            raise ValueError("opponent weights must sum to > 0")
        return w / w.sum()

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def current_mix(self) -> dict[str, float]:
        """Public for logging/inspection."""
        weights = self._weights_at(self.current_step)
        return dict(zip(self.names, weights.tolist()))

    def _weights_at(self, step: int) -> np.ndarray:
        if step <= self.schedule_start_step:
            t = 0.0
        elif step >= self.schedule_end_step:
            t = 1.0
        else:
            t = (step - self.schedule_start_step) / (
                self.schedule_end_step - self.schedule_start_step
            )
        return (1.0 - t) * self.start_weights + t * self.end_weights

    def sample(self, rng):
        probs = self._weights_at(self.current_step)
        idx = int(rng.choice(len(self.names), p=probs))
        name = self.names[idx]
        if name == self.SELF_KEY:
            return self._self_factory()
        return self._builtin_factories[name]

    def _self_factory(self) -> OpponentFactory:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return self.fallback_factory
        rep = self.representation
        snapshot_path = self.snapshot_path
        enable_maritime = self.enable_maritime_trade

        def _factory(color: Color) -> Player:
            from catan_rl.players.rl_player import RLPlayer

            return RLPlayer(
                color, snapshot_path, representation=rep, deterministic=False,
                enable_maritime_trade=enable_maritime,
            )

        return _factory


class EpsilonPlayer(Player):
    """Wraps a base bot; with probability epsilon plays a uniform random legal
    move instead. epsilon=1 ~ random, epsilon=0 = the full bot — a continuous
    difficulty dial between the two."""

    def __init__(self, color: Color, base: Player, epsilon: float, seed: int | None = None):
        super().__init__(color)
        self.base = base
        self.epsilon = float(epsilon)
        import random as _random
        self._rng = _random.Random(seed)

    def decide(self, game, playable_actions):
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(playable_actions))
        return self.base.decide(game, playable_actions)


class EpsilonCurriculumOpponent(OpponentSampler):
    """Fixed-weight mix whose main rung is an epsilon-handicapped bot whose
    epsilon is auto-calibrated externally (eval callback pushes updates via
    `set_epsilon` so the agent's winrate vs it stays ~target).

      self_weight   -> best-past self snapshot
      eps_weight    -> EpsilonPlayer(eps_base bot, current epsilon)
      floor_mix     -> small fixed exposure to the real hard bots (e.g. value/ab1)
    """

    SELF_KEY = "self"
    EPS_KEY = "eps_value"

    def __init__(
        self,
        *,
        eps_base: str = "value",
        eps_init: float = 0.75,
        self_weight: float = 0.5,
        eps_weight: float = 0.35,
        floor_mix: Mapping[str, float] | None = None,
        snapshot_path: Path | str | None = None,
        fallback_name: str = "weighted",
        representation: str = "mixed",
        enable_maritime_trade: bool = False,
    ):
        self.eps_base = eps_base
        self.epsilon = float(eps_init)
        self.self_weight = float(self_weight)
        self.eps_weight = float(eps_weight)
        self.floor_mix = dict(floor_mix or {})
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self.fallback_name = fallback_name
        self.fallback_factory = _builtin_factory(fallback_name)
        self.representation = representation
        self.enable_maritime_trade = enable_maritime_trade
        self._base_factory = _builtin_factory(eps_base)
        self._floor_factories = {n: _builtin_factory(n) for n in self.floor_mix}
        self.current_step = 0
        self._eps_seed = 0

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def set_epsilon(self, eps: float) -> None:
        self.epsilon = float(min(1.0, max(0.0, eps)))

    def current_mix(self) -> dict[str, float]:
        m = {self.SELF_KEY: self.self_weight, self.EPS_KEY: self.eps_weight}
        m.update(self.floor_mix)
        m["epsilon"] = self.epsilon  # logged alongside weights in W&B
        return m

    def sample(self, rng):
        names = [self.SELF_KEY, self.EPS_KEY] + list(self.floor_mix)
        probs = np.array(
            [self.self_weight, self.eps_weight] + list(self.floor_mix.values()),
            dtype=float,
        )
        probs = probs / probs.sum()
        name = names[int(rng.choice(len(names), p=probs))]
        if name == self.SELF_KEY:
            return self._self_factory()
        if name == self.EPS_KEY:
            eps, base_f = self.epsilon, self._base_factory
            self._eps_seed += 1
            seed = self._eps_seed

            def _factory(color: Color) -> Player:
                return EpsilonPlayer(color, base_f(color), eps, seed=seed)
            return _factory
        return self._floor_factories[name]

    def _self_factory(self) -> OpponentFactory:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return self.fallback_factory
        rep, snap, mar = self.representation, self.snapshot_path, self.enable_maritime_trade

        def _factory(color: Color) -> Player:
            from catan_rl.players.rl_player import RLPlayer
            return RLPlayer(color, snap, representation=rep, deterministic=False,
                            enable_maritime_trade=mar)
        return _factory


class AdaptiveOpponent(OpponentSampler):
    """Adaptive curriculum over a difficulty `ladder` (e.g. weighted -> value ->
    alphabeta1 -> alphabeta2), driven by recent eval winrates.

    Budget per episode:
      - `self_weight` (default 0.5) -> the self snapshot (best-past; always ~winnable).
      - the rest -> the <=2 lowest ladder rungs not yet "mastered"
        (winrate >= `master_threshold`), split by contestability `w*(1-w)+eps`,
        so the sweet-spot (~0.5) rung gets the most signal.
      - `floor_weight` -> a small exposure to the next-harder rung above the
        active window (so the agent always sees a bit of what's coming).

    Winrates are pushed in from the eval callback via `set_winrates`
    (env_method -> wrapper -> here). Cold start (no winrates) = first two rungs.
    """

    SELF_KEY = "self"
    _EPS = 0.10  # keeps an unbeaten active rung from getting ~0 weight

    def __init__(
        self,
        ladder: list[str],
        *,
        self_weight: float = 0.5,
        floor_weight: float = 0.05,
        master_threshold: float = 0.85,
        snapshot_path: Path | str | None = None,
        fallback_name: str = "weighted",
        representation: str = "mixed",
        enable_maritime_trade: bool = False,
    ):
        self.ladder = list(ladder)
        self.self_weight = float(self_weight)
        self.floor_weight = float(floor_weight)
        self.master_threshold = float(master_threshold)
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self.fallback_name = fallback_name
        self.fallback_factory = _builtin_factory(fallback_name)
        self.representation = representation
        self.enable_maritime_trade = enable_maritime_trade
        self._winrates: dict[str, float] = {}
        self.current_step = 0
        self._builtin_factories = {n: _builtin_factory(n) for n in self.ladder}

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def set_winrates(self, winrates: Mapping[str, float]) -> None:
        self._winrates.update({k: float(v) for k, v in winrates.items()})

    def _mix(self) -> dict[str, float]:
        non_self = max(0.0, 1.0 - self.self_weight)
        w = lambda r: self._winrates.get(r, 0.0)

        # Cold start (no eval data yet): w*(1-w) is 0 for every rung, so it can't
        # tell the easy rung from the hard one. Focus the easiest rung and put the
        # next one in the floor; once real winrates arrive, w*(1-w) takes over.
        if not self._winrates:
            if len(self.ladder) == 1:
                return {self.SELF_KEY: self.self_weight, self.ladder[0]: non_self}
            return {self.SELF_KEY: self.self_weight,
                    self.ladder[0]: non_self - self.floor_weight,
                    self.ladder[1]: self.floor_weight}

        # Measured: w*(1-w) over the <=2 easiest unmastered rungs + floor on the next.
        non_mastered = [r for r in self.ladder if w(r) < self.master_threshold]
        if not non_mastered:                       # everything mastered -> drill the top
            return {self.SELF_KEY: self.self_weight, self.ladder[-1]: non_self}
        active = non_mastered[:2]
        floor_rung = non_mastered[2] if len(non_mastered) > 2 else None

        mix = {self.SELF_KEY: self.self_weight}
        floor = self.floor_weight if floor_rung is not None else 0.0
        budget = non_self - floor
        scores = {r: w(r) * (1 - w(r)) + self._EPS for r in active}
        tot = sum(scores.values())
        for r in active:
            mix[r] = budget * scores[r] / tot
        if floor_rung is not None:
            mix[floor_rung] = mix.get(floor_rung, 0.0) + floor
        return mix

    def current_mix(self) -> dict[str, float]:
        return self._mix()

    def sample(self, rng):
        mix = self._mix()
        names = list(mix.keys())
        probs = np.array([mix[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        name = names[int(rng.choice(len(names), p=probs))]
        if name == self.SELF_KEY:
            return self._self_factory()
        return self._builtin_factories[name]

    def _self_factory(self) -> OpponentFactory:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return self.fallback_factory
        rep, snap, enable_mar = self.representation, self.snapshot_path, self.enable_maritime_trade

        def _factory(color: Color) -> Player:
            from catan_rl.players.rl_player import RLPlayer
            return RLPlayer(color, snap, representation=rep, deterministic=False,
                            enable_maritime_trade=enable_mar)
        return _factory


def build_sampler(opponent_cfg, *, snapshot_path: Path | str | None = None,
                  representation: str = "mixed",
                  enable_maritime_trade: bool = False) -> OpponentSampler:
    """Resolve an `OpponentConfig` to a concrete `OpponentSampler`."""
    if opponent_cfg.sampler == "single":
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
            enable_maritime_trade=enable_maritime_trade,
        )
    if opponent_cfg.sampler == "scheduled":
        if snapshot_path is None:
            raise ValueError("scheduled sampler requires snapshot_path (for 'self' slot)")
        return ScheduledOpponent(
            start_mix=dict(opponent_cfg.start_mix),
            end_mix=dict(opponent_cfg.end_mix),
            schedule_start_step=opponent_cfg.schedule_start_step,
            schedule_end_step=opponent_cfg.schedule_end_step,
            snapshot_path=snapshot_path,
            fallback_name=opponent_cfg.self_fallback,
            representation=representation,
            enable_maritime_trade=enable_maritime_trade,
        )
    if opponent_cfg.sampler == "epsilon":
        if snapshot_path is None:
            raise ValueError("epsilon sampler requires snapshot_path (for 'self' slot)")
        return EpsilonCurriculumOpponent(
            eps_base=opponent_cfg.eps_base,
            eps_init=opponent_cfg.eps_init,
            self_weight=opponent_cfg.self_weight,
            eps_weight=opponent_cfg.eps_weight,
            floor_mix=dict(opponent_cfg.eps_floor_mix),
            snapshot_path=snapshot_path,
            fallback_name=opponent_cfg.self_fallback,
            representation=representation,
            enable_maritime_trade=enable_maritime_trade,
        )
    if opponent_cfg.sampler == "adaptive":
        if snapshot_path is None:
            raise ValueError("adaptive sampler requires snapshot_path (for 'self' slot)")
        return AdaptiveOpponent(
            list(opponent_cfg.ladder),
            self_weight=opponent_cfg.self_weight,
            floor_weight=opponent_cfg.floor_weight,
            master_threshold=opponent_cfg.master_threshold,
            snapshot_path=snapshot_path,
            fallback_name=opponent_cfg.self_fallback,
            representation=representation,
            enable_maritime_trade=enable_maritime_trade,
        )
    raise ValueError(f"unknown sampler {opponent_cfg.sampler!r}")
