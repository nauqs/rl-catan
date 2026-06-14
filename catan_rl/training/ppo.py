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
from catan_rl.utils.config import BaseConfig


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
    sampler = build_sampler(
        cfg.train.opponent_cfg,
        snapshot_path=run_dir / "self_snapshot.zip",
        representation=cfg.env.representation,
        enable_maritime_trade=cfg.env.enable_maritime_trade,
    )
    # Epsilon curriculum: resume from the persisted calibrated epsilon if present.
    eps_file = run_dir / "epsilon.txt"
    if hasattr(sampler, "set_epsilon") and eps_file.exists():
        try:
            sampler.set_epsilon(float(eps_file.read_text().strip()))
        except ValueError:
            pass
    return sampler


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_lr_schedule(cfg: BaseConfig):
    """Constant LR, or linear warmup 0->lr over the first `warmup_steps` then hold.
    SB3 passes progress_remaining (1 at start -> 0 at end); convert to env steps."""
    lr_peak = cfg.train.ppo.learning_rate
    warmup = cfg.train.ppo.warmup_steps
    total = cfg.train.total_timesteps
    if warmup and warmup > 0:
        def lr_schedule(progress_remaining: float) -> float:
            step = total * (1.0 - progress_remaining)
            return lr_peak * min(1.0, step / warmup)
        return lr_schedule
    return lr_peak


def _apply_ppo_hypers(model, cfg: BaseConfig) -> None:
    """Re-apply the config's PPO hypers to a *loaded* model.

    `MaskablePPO.load` restores the hyperparameters baked into the checkpoint. For
    a BC seed those are the BC-era values (ent 0.01 / clip 0.2 / lr 1e-4), so a
    resume would silently ignore a new RL recipe (e.g. the IIG regime: ent 0.1,
    clip 0.025, vf 0.06). Override the safe-to-change ones here. (gamma / gae /
    n_steps / batch_size are tied to the rollout buffer and are intentionally left
    at the checkpoint's values, which already match our intent.)"""
    from stable_baselines3.common.utils import get_schedule_fn
    p = cfg.train.ppo
    model.ent_coef = p.ent_coef
    model.vf_coef = p.vf_coef
    model.max_grad_norm = p.max_grad_norm
    model.n_epochs = p.n_epochs
    model.clip_range = get_schedule_fn(p.clip_range)
    lr = _make_lr_schedule(cfg)
    model.lr_schedule = lr if callable(lr) else get_schedule_fn(lr)
    model.learning_rate = p.learning_rate
    if model.verbose:
        print(f"[hypers] resume override: ent_coef={p.ent_coef} clip={p.clip_range} "
              f"vf={p.vf_coef} lr={p.learning_rate} n_epochs={p.n_epochs} "
              f"warmup={p.warmup_steps}")


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
                enable_maritime_trade=cfg.env.enable_maritime_trade,
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
    from catan_rl.utils.keep_awake import keep_awake
    keep_awake()  # prevent macOS idle-sleep from pausing the run (no-op elsewhere)

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

    if cfg.train.policy == "spatial":
        from catanatron.models.player import Color
        from catan_rl.features import HexEncoder, HexFormer, XdimTrunk
        from catan_rl.policies.spatial import SpatialMaskableActorCriticPolicy
        policy_cls = SpatialMaskableActorCriticPolicy
        # Trunk feeds the shared spatial pointer head. "conv" = HexEncoder
        # (Phase 3); "hexformer" = adjacency-aware Tile Transformer (Phase 4);
        # "xdim" = cross-dimensional board<->scalar trunk (Gendre-Kaneko 2020).
        trunk_cls = {"hexformer": HexFormer, "xdim": XdimTrunk}.get(
            cfg.train.trunk, HexEncoder
        )
        # The trunk owns the feature map + global summary; the spatial head owns
        # the policy/value MLPs, so we don't pass net_arch / features_dim here.
        policy_kwargs = dict(
            features_extractor_class=trunk_cls,
            player_colors=(Color.BLUE, Color.RED),
            map_type=cfg.env.map_type,
            enable_maritime=cfg.env.enable_maritime_trade,
        )
    elif cfg.train.policy == "factorized":
        from catanatron.models.player import Color
        from catan_rl.policies.factorized import FactorizedMaskableActorCriticPolicy
        policy_cls = FactorizedMaskableActorCriticPolicy
        # 1v1: trainable agent is BLUE, single enemy is RED. Phase 2's
        # mixed-opponent sampler keeps the *color* tuple stable across opponents.
        policy_kwargs = dict(
            features_extractor_class=EXTRACTORS[cfg.train.arch],
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            player_colors=(Color.BLUE, Color.RED),
            map_type=cfg.env.map_type,
        )
    else:
        policy_cls = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=EXTRACTORS[cfg.train.arch],
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

    # KL-anchor / MMD: when on, train with the KLAnchoredMaskablePPO subclass so the
    # policy is regularized toward a (possibly moving) magnet + an MMD proximal term.
    p = cfg.train.ppo
    kl_coef = p.kl_coef
    # Use the MMD subclass when ANY of its extra mechanisms are on: BC magnet
    # (kl_coef>0), a moving magnet, the MMD proximal term, or entropy annealing.
    # The last three don't need a BC reference (works from scratch).
    use_anchor = (
        (p.kl_coef and p.kl_coef > 0)
        or (p.mmd_kl_coef and p.mmd_kl_coef > 0)
        or (p.magnet_mode != "fixed")
        or (p.ent_coef_final is not None)
    )
    ppo_cls = MaskablePPO
    if use_anchor:
        from catan_rl.training.kl_anchor import KLAnchoredMaskablePPO
        ppo_cls = KLAnchoredMaskablePPO

    if resume and (run_dir / "model.zip").exists():
        model = ppo_cls.load(
            str(run_dir / "model.zip"),
            env=venv,
            device=cfg.train.device,
            # Force logging config regardless of what the checkpoint was saved
            # with (e.g. a BC-pretrained seed model built without tensorboard).
            custom_objects={"tensorboard_log": str(run_dir / "tb"), "verbose": 1},
        )
        # Resume restores the checkpoint's hypers; re-apply the config recipe so a
        # BC seed picks up the RL hypers (IIG regime) rather than the BC-era ones.
        _apply_ppo_hypers(model, cfg)
    else:
        learning_rate = _make_lr_schedule(cfg)
        model = ppo_cls(
            policy_cls,
            venv,
            gamma=cfg.train.ppo.gamma,
            gae_lambda=cfg.train.ppo.gae_lambda,
            n_steps=cfg.train.ppo.n_steps,
            batch_size=cfg.train.ppo.batch_size,
            ent_coef=cfg.train.ppo.ent_coef,
            learning_rate=learning_rate,
            n_epochs=cfg.train.ppo.n_epochs,
            clip_range=cfg.train.ppo.clip_range,
            vf_coef=cfg.train.ppo.vf_coef,
            max_grad_norm=cfg.train.ppo.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tb"),
            device=cfg.train.device,
            verbose=1,
        )

    # Enable the KL-to-BC anchor. The reference is the BC seed: capture it once
    # into run_dir/bc_ref.zip on the first launch (model.zip is still the freshly
    # loaded seed here, before any RL save), then reuse it across resumes so the
    # anchor never drifts to a mid-RL checkpoint. kl_ref_path overrides the source.
    if ppo_cls is not MaskablePPO:
        import shutil
        # Only materialize a reference when the magnet actually needs one
        # (BC KL term or a moving magnet). MMD-proximal / ent-anneal need no ref.
        needs_ref = (kl_coef and kl_coef > 0) or p.magnet_mode != "fixed"
        ref_path = None
        if needs_ref:
            if cfg.train.ppo.kl_ref_path:
                ref_path = Path(cfg.train.ppo.kl_ref_path)
            else:
                ref_path = run_dir / "bc_ref.zip"
                if not ref_path.exists() and (run_dir / "model.zip").exists():
                    shutil.copyfile(run_dir / "model.zip", ref_path)
        model.set_kl_anchor(
            str(ref_path) if ref_path is not None else None, kl_coef, cfg.train.ppo.kl_target,
            magnet_mode=p.magnet_mode,
            magnet_ema=p.magnet_ema,
            magnet_reset_steps=p.magnet_reset_steps,
            mmd_kl_coef=p.mmd_kl_coef,
            ent_coef_final=p.ent_coef_final,
            total_timesteps=cfg.train.total_timesteps,
            magnet_state_path=str(run_dir / "magnet.pt"),
        )

    is_adaptive = (cfg.train.opponent_cfg is not None
                   and cfg.train.opponent_cfg.sampler in ("adaptive", "epsilon"))
    best_past = is_adaptive and cfg.train.opponent_cfg.best_past_self
    eps_cfg = None
    if (cfg.train.opponent_cfg is not None
            and cfg.train.opponent_cfg.sampler == "epsilon"):
        oc = cfg.train.opponent_cfg
        eps_cfg = dict(base=oc.eps_base, target=oc.eps_target, lr=oc.eps_lr,
                       init=oc.eps_init, games=oc.eps_games)

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
            enable_maritime_trade=cfg.env.enable_maritime_trade,
            snapshot_path=(run_dir / "self_snapshot.zip") if best_past else None,
            best_past_self=best_past,
            eps_cfg=eps_cfg,
        ),
    ]
    # If MixedWithSelf or Scheduled (with 'self' in the mix) is configured,
    # periodically snapshot the policy so the 'self' slot can load it.
    # (Adaptive uses MiniLadderCallback's best-past snapshot instead.)
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
    # Scheduled/Adaptive/Epsilon samplers all want per-rollout step push + mix logging.
    if (cfg.train.opponent_cfg is not None
            and cfg.train.opponent_cfg.sampler in ("scheduled", "adaptive", "epsilon")):
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
            color, model_path, representation=cfg.env.representation, deterministic=True,
            enable_maritime_trade=cfg.env.enable_maritime_trade,
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
