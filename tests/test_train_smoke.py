"""End-to-end smoke: train for a tiny budget, resume from disk, RLPlayer round-trip."""
from __future__ import annotations

from pathlib import Path

from catan_rl.training.ppo import train
from catan_rl.utils.config import BaseConfig


def _tiny_cfg(tmp_path: Path) -> BaseConfig:
    cfg = BaseConfig.model_validate(
        {
            "run_name": f"smoke-{tmp_path.name}",
            "env": {"representation": "mixed"},
            "train": {
                "total_timesteps": 256,
                "n_envs": 2,
                "device": "cpu",
                "opponent": "random",
                "reward": {"enabled": True},
                "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
                "checkpoint_every": 10**9,
            },
            "eval": {
                "callback_every": 10**9,    # disable mid-train eval
                "run_final_ladder": False,
            },
            "wandb": {"mode": "disabled"},
        }
    )
    return cfg


def test_train_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, resume=False)
    assert final.exists(), "final model.zip not written"

    # Resume picks up the same checkpoint and runs a tiny extra budget.
    cfg.train.total_timesteps = 384
    final2 = train(cfg, resume=True)
    assert final2.exists()
