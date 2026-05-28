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
    ScheduledOpponent,
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


# ---------- ScheduledOpponent ----------

def test_scheduled_interpolates_linearly(tmp_path: Path) -> None:
    s = ScheduledOpponent(
        start_mix={"random": 1.0},
        end_mix={"weighted": 1.0},
        schedule_start_step=100,
        schedule_end_step=200,
        snapshot_path=tmp_path / "snap.zip",
    )
    # Before start: 100% random
    s.set_step(0)
    rng = np.random.default_rng(0)
    samples = [type(s.sample(rng)(Color.RED)).__name__ for _ in range(200)]
    assert all(n == "RandomPlayer" for n in samples)

    # After end: 100% weighted
    s.set_step(300)
    samples = [type(s.sample(rng)(Color.RED)).__name__ for _ in range(200)]
    assert all(n == "WeightedRandomPlayer" for n in samples)

    # Midpoint (step=150): expect ~50/50
    s.set_step(150)
    counts = {"RandomPlayer": 0, "WeightedRandomPlayer": 0}
    for _ in range(2000):
        n = type(s.sample(rng)(Color.RED)).__name__
        counts[n] += 1
    frac_random = counts["RandomPlayer"] / 2000
    assert 0.45 < frac_random < 0.55, f"expected ~0.5 random, got {frac_random}"


def test_scheduled_self_slot_falls_back_when_no_snapshot(tmp_path: Path) -> None:
    s = ScheduledOpponent(
        start_mix={"weighted": 0.5, "self": 0.5},
        end_mix={"weighted": 0.5, "self": 0.5},
        schedule_start_step=0,
        schedule_end_step=1,
        snapshot_path=tmp_path / "no_such.zip",
        fallback_name="random",
    )
    s.set_step(0)
    rng = np.random.default_rng(0)
    types = [type(s.sample(rng)(Color.RED)).__name__ for _ in range(200)]
    # Self slot falls back to RandomPlayer; weighted slot returns WeightedRandomPlayer.
    assert "RandomPlayer" in types
    assert "WeightedRandomPlayer" in types


def test_scheduled_current_mix_logging() -> None:
    s = ScheduledOpponent(
        start_mix={"random": 0.8, "weighted": 0.2},
        end_mix={"value": 0.5, "self": 0.5},
        schedule_start_step=0,
        schedule_end_step=100,
        snapshot_path=Path("/tmp/snap.zip"),
    )
    s.set_step(0)
    mix0 = s.current_mix()
    assert mix0["random"] == pytest.approx(0.8)
    assert mix0["weighted"] == pytest.approx(0.2)
    assert mix0["value"] == pytest.approx(0.0)
    assert mix0["self"] == pytest.approx(0.0)

    s.set_step(100)
    mix1 = s.current_mix()
    assert mix1["random"] == pytest.approx(0.0)
    assert mix1["weighted"] == pytest.approx(0.0)
    assert mix1["value"] == pytest.approx(0.5)
    assert mix1["self"] == pytest.approx(0.5)


def test_build_sampler_dispatches_scheduled(tmp_path: Path) -> None:
    cfg = OpponentConfig(
        sampler="scheduled",
        start_mix={"random": 1.0},
        end_mix={"weighted": 1.0},
        schedule_start_step=0,
        schedule_end_step=10,
    )
    s = build_sampler(cfg, snapshot_path=tmp_path / "snap.zip")
    assert isinstance(s, ScheduledOpponent)
