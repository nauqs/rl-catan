"""OpponentSampler: distributions converge to mix; MixedWithSelf falls back when no snapshot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catanatron import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.training.opponents import (
    MixedOpponent,
    MixedWithSelf,
    SingleOpponent,
    build_sampler,
)
from catan_rl.utils.config import OpponentConfig


def test_single_opponent_always_returns_same_class() -> None:
    s = SingleOpponent("weighted")
    rng = np.random.default_rng(0)
    for _ in range(5):
        f = s.sample(rng)
        assert isinstance(f(Color.RED), WeightedRandomPlayer)


def test_mixed_opponent_converges_to_weights() -> None:
    mix = {"random": 0.7, "value": 0.3}
    s = MixedOpponent(mix)
    rng = np.random.default_rng(0)
    counts = {"random": 0, "value": 0}
    for _ in range(2000):
        p = s.sample(rng)(Color.RED)
        if isinstance(p, RandomPlayer):
            counts["random"] += 1
        elif isinstance(p, ValueFunctionPlayer):
            counts["value"] += 1
    total = sum(counts.values())
    # ±3% tolerance over 2000 samples.
    assert abs(counts["random"] / total - 0.7) < 0.03
    assert abs(counts["value"] / total - 0.3) < 0.03


def test_mixed_with_self_falls_back_when_no_snapshot(tmp_path: Path) -> None:
    mix = {"weighted": 0.5, "self": 0.5}
    snap = tmp_path / "snap.zip"  # does not exist
    s = MixedWithSelf(mix, snapshot_path=snap, fallback_name="random")
    rng = np.random.default_rng(0)
    seen = {"weighted": 0, "self_fallback_random": 0, "other": 0}
    for _ in range(200):
        p = s.sample(rng)(Color.RED)
        if isinstance(p, WeightedRandomPlayer):
            seen["weighted"] += 1
        elif isinstance(p, RandomPlayer):
            seen["self_fallback_random"] += 1
        else:
            seen["other"] += 1
    # Roughly half of samples should hit the self slot, which falls back to random.
    assert seen["self_fallback_random"] > 0
    assert seen["weighted"] > 0
    assert seen["other"] == 0


def test_build_sampler_dispatches_correctly(tmp_path: Path) -> None:
    cfg_single = OpponentConfig(sampler="single", mix={"weighted": 1.0})
    assert isinstance(build_sampler(cfg_single), SingleOpponent)

    cfg_mixed = OpponentConfig(sampler="mixed", mix={"random": 0.5, "weighted": 0.5})
    assert isinstance(build_sampler(cfg_mixed), MixedOpponent)

    cfg_self = OpponentConfig(
        sampler="mixed_with_self", mix={"weighted": 0.5, "self": 0.5}
    )
    s = build_sampler(cfg_self, snapshot_path=tmp_path / "snap.zip")
    assert isinstance(s, MixedWithSelf)


def test_mixed_with_self_requires_self_key() -> None:
    with pytest.raises(ValueError):
        MixedWithSelf({"weighted": 1.0}, snapshot_path="/tmp/x")
