"""MaskablePPO training loop with W&B + crash-safe resume + Phase-2a modules."""
from __future__ import annotations

import json
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from catan_rl.features import EXTRACTORS
from catan_rl.training.callbacks import (
    AnnealingCallback,
    MiniLadderCallback,
    OpponentScheduleCallback,
    SelfSnapshotCallback,
)
from catan_rl.training.env_factory import make_env
from catan_rl.training.opponents import (
    OpponentSampler,
    SingleOpponent,
    build_sampler,
)
from catan_rl.utils.config import BaseConfig, OpponentConfig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _resolve_run_dir(cfg: BaseConfig) -> Path:
    phase = "phase2" if cfg.train.opponent_cfg is not None else "phase1"
    return Path("checkpoints") / phase / cfg.run_name


def _resolve_sampler(cfg: BaseConfig, run_dir: Path) -> OpponentSampler:
    """Build the OpponentSampler from cfg, falling back to Phase-1 single-opponent."""
    if cfg.train.opponent_cfg is None:
        return SingleOpponent(cfg.train.opponent)
    return build_sampler(
        cfg.train.opponent_cfg,
        snapshot_path=run_dir / "self_snapshot.zip",
        representation=cfg.env.representation,
    )


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_vec_env(cfg: BaseConfig, run_dir: Path) -> SubprocVecEnv:
    n = cfg.train.n_envs

    def _factory(rank: int):
        # Re-resolve the sampler *inside the subprocess* (it gets pickled cleanly).
        sampler = _resolve_sampler(cfg, run_dir)

        def _thunk():
            return make_env(
                seed=cfg.seed + rank,
                map_type=cfg.env.map_type,
                representation=cfg.env.representation,
                sampler=sampler,
                reward_shaping=cfg.train.reward.enabled,
                reward_config=cfg.train.reward,
            )

        return _thunk

    return SubprocVecEnv([_factory(i) for i in range(n)])


class _UpdateLatestCallback(BaseCallback):
    """After each PPO rollout, overwrite `model.zip` so resume has a single
    canonical path to load from."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.save(str(self.run_dir / "model.zip"))
        _save_text(self.run_dir / "step.txt", str(self.num_timesteps))


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def train(cfg: BaseConfig, *, resume: bool = False) -> Path:
    """Train MaskablePPO with the given config. Returns path to final `model.zip`."""
    run_dir = _resolve_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = _make_vec_env(cfg, run_dir)

    wandb_run = None
    wandb_callback = None
    if cfg.wandb.mode != "disabled":
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run_id_path = run_dir / "wandb_run_id.txt"
        common_init = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or os.environ.get("WANDB_ENTITY"),
            mode=cfg.wandb.mode,
            config=cfg.model_dump(),
            sync_tensorboard=True,
        )
        if resume and run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            wandb_run = wandb.init(id=run_id, resume="must", **common_init)
        else:
            run_id = wandb.util.generate_id()
            wandb_run = wandb.init(id=run_id, name=cfg.run_name, **common_init)
            run_id_path.write_text(run_id)
        wandb_callback = WandbCallback(verbose=0)

    extractor_cls = EXTRACTORS[cfg.train.arch]
    policy_kwargs = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if cfg.train.policy == "factorized":
        from catanatron.models.player import Color
        from catan_rl.policies.factorized import FactorizedMaskableActorCriticPolicy
        policy_cls = FactorizedMaskableActorCriticPolicy
        # 1v1: trainable agent is BLUE, single enemy is RED. Phase 2's
        # mixed-opponent sampler keeps the *color* tuple stable across opponents.
        policy_kwargs["player_colors"] = (Color.BLUE, Color.RED)
        policy_kwargs["map_type"] = cfg.env.map_type
    else:
        policy_cls = "MultiInputPolicy"

    if resume and (run_dir / "model.zip").exists():
        model = MaskablePPO.load(
            str(run_dir / "model.zip"),
            env=venv,
            device=cfg.train.device,
        )
    else:
        model = MaskablePPO(
            policy_cls,
            venv,
            gamma=cfg.train.ppo.gamma,
            gae_lambda=cfg.train.ppo.gae_lambda,
            n_steps=cfg.train.ppo.n_steps,
            batch_size=cfg.train.ppo.batch_size,
            ent_coef=cfg.train.ppo.ent_coef,
            learning_rate=cfg.train.ppo.learning_rate,
            n_epochs=cfg.train.ppo.n_epochs,
            clip_range=cfg.train.ppo.clip_range,
            vf_coef=cfg.train.ppo.vf_coef,
            max_grad_norm=cfg.train.ppo.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb"),
            device=cfg.train.device,
            verbose=1,
        )

    save_freq = max(1, cfg.train.checkpoint_every // cfg.train.n_envs)
    callbacks: list[BaseCallback] = [
        _UpdateLatestCallback(run_dir),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(run_dir / "ckpts"),
            name_prefix="model",
            verbose=0,
        ),
        AnnealingCallback(
            start_step=cfg.train.reward.anneal_start_step,
            end_step=cfg.train.reward.anneal_end_step,
        ),
        MiniLadderCallback(
            eval_every=cfg.eval.callback_every,
            opponents=cfg.eval.callback_opponents,
            n_games=cfg.eval.callback_games,
            save_dir=run_dir,
            representation=cfg.env.representation,
        ),
    ]
    # If MixedWithSelf or Scheduled (with 'self' in the mix) is configured,
    # periodically snapshot the policy so the 'self' slot can load it.
    if cfg.train.opponent_cfg is not None and cfg.train.opponent_cfg.sampler in (
        "mixed_with_self", "scheduled",
    ):
        wants_self = (
            "self" in cfg.train.opponent_cfg.mix
            or "self" in cfg.train.opponent_cfg.start_mix
            or "self" in cfg.train.opponent_cfg.end_mix
        )
        if wants_self:
            callbacks.append(SelfSnapshotCallback(
                snapshot_path=run_dir / "self_snapshot.zip",
                snapshot_every=cfg.train.opponent_cfg.snapshot_every_steps,
            ))
    # ScheduledOpponent needs to know the current step to interpolate.
    if (cfg.train.opponent_cfg is not None
            and cfg.train.opponent_cfg.sampler == "scheduled"):
        callbacks.append(OpponentScheduleCallback())
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    try:
        model.learn(
            total_timesteps=cfg.train.total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not resume,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[train] KeyboardInterrupt — saving and exiting.")
    finally:
        model.save(str(run_dir / "model.zip"))
        _save_text(run_dir / "step.txt", str(model.num_timesteps))
        venv.close()

    final_path = run_dir / "model.zip"
    if cfg.eval.run_final_ladder and model.num_timesteps >= cfg.train.total_timesteps:
        _run_final_ladder(cfg, final_path, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    return final_path


def _run_final_ladder(cfg: BaseConfig, model_path: Path, wandb_run) -> None:
    from catan_rl.eval.ladder import run_ladder
    from catan_rl.players.rl_player import RLPlayer

    def factory(color):
        return RLPlayer(
            color, model_path, representation=cfg.env.representation, deterministic=True
        )

    result = run_ladder(
        factory,
        candidate_name=cfg.run_name,
        opponents=cfg.eval.final_opponents,
        n_games=cfg.eval.final_games,
        modes=tuple(cfg.eval.final_modes),
    )
    payload = result.to_dict()
    out_path = model_path.parent / "final_ladder.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if wandb_run is not None:
        import wandb

        for row in payload["results"]:
            wandb.summary[f"final/{row['mode']}/{row['opponent']}/winrate"] = row["winrate"]
            wandb.summary[f"final/{row['mode']}/{row['opponent']}/mean_vp"] = row["mean_vp"]
        wandb.summary["final/results"] = payload["results"]
