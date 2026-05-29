"""PPO callbacks: dense-reward annealing + lightweight in-training ladder eval."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class AnnealingCallback(BaseCallback):
    """Linearly tapers the dense-reward factor from 1.0 -> 0.0 between two env-step bounds.

    No-op if either bound is < 0 (Charlesworth's default). Reads `num_timesteps`
    from the SB3 model and pushes the current factor into every env via
    `training_env.env_method("set_dense_factor", ...)`.
    """

    def __init__(self, start_step: int, end_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step

    @property
    def enabled(self) -> bool:
        return self.start_step >= 0 and self.end_step > self.start_step

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        n = self.num_timesteps
        if n <= self.start_step:
            factor = 1.0
        elif n >= self.end_step:
            factor = 0.0
        else:
            span = self.end_step - self.start_step
            factor = 1.0 + (n - self.start_step) / span * (0.0 - 1.0)
        self.training_env.env_method("set_dense_factor", factor)
        self.logger.record("rollout/dense_factor", factor)
        return True


class MiniLadderCallback(BaseCallback):
    """Every `eval_every` env steps, run a cheap ladder eval and log per-opponent
    winrate to W&B / TB. Tracks best vs-Random winrate and copies that checkpoint
    to `<save_dir>/best.zip`.
    """

    def __init__(
        self,
        eval_every: int,
        opponents: list[str],
        n_games: int,
        save_dir: Path,
        representation: str = "mixed",
        enable_maritime_trade: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.opponents = opponents
        self.n_games = n_games
        self.save_dir = Path(save_dir)
        self.representation = representation
        self.enable_maritime_trade = enable_maritime_trade
        self._last_eval_step = 0
        self._best_random_wr: float = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_every:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        from catan_rl.eval.ladder import run_ladder
        from catan_rl.eval.metrics import aggregate, collect
        from catan_rl.eval.opponents import build_opponent
        from catan_rl.players.rl_player import RLPlayer
        from catanatron import Color, Game

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "current.zip"
            self.model.save(str(ckpt))

            # Ladder winrate: deterministic (stable winrate). Behavioral metrics
            # below: stochastic, so action-tendency counts (robber/maritime/dev)
            # reflect the policy's distribution rather than a single argmax choice.
            def factory(color, deterministic=True):
                return RLPlayer(
                    color, ckpt, representation=self.representation,
                    deterministic=deterministic,
                    enable_maritime_trade=self.enable_maritime_trade,
                )

            result = run_ladder(
                factory,
                candidate_name=f"step_{self.num_timesteps}",
                opponents=self.opponents,
                n_games=self.n_games,
                modes=("1v1",),
            )

            random_wr = 0.0
            for row in result.by_opponent:
                self.logger.record(f"eval/1v1/{row.opponent}/winrate", row.winrate)
                self.logger.record(f"eval/1v1/{row.opponent}/mean_vp", row.mean_vp)
                self.logger.record(f"eval/1v1/{row.opponent}/mean_turns", row.mean_turns)
                if row.opponent == "random":
                    random_wr = row.winrate

            if random_wr > self._best_random_wr:
                self._best_random_wr = random_wr
                self.save_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt, self.save_dir / "best.zip")
                self.logger.record("eval/best_random_wr", random_wr)

            # --- cheap behavioral metrics (10 stochastic games vs weighted) ---
            beh_rows = []
            for s in range(10):
                game = Game(
                    [factory(Color.BLUE, deterministic=False),
                     build_opponent("weighted", Color.RED)],
                    seed=self.num_timesteps + s,
                )
                game.play()
                beh_rows.append(collect(game, Color.BLUE))
            agg = aggregate(beh_rows)
            for k, v in agg.items():
                self.logger.record(f"behavior/{k}", v)

        if self.verbose:
            print(f"[MiniLadder] step={self.num_timesteps} best_random_wr={self._best_random_wr:.2f}")


class OpponentScheduleCallback(BaseCallback):
    """Pushes `num_timesteps` to every env's opponent sampler so a
    `ScheduledOpponent` can compute the current interpolated mix. No-op for
    non-scheduled samplers (their `set_step` is a no-op).

    Fires on every rollout start (cheap; happens N times per training run
    where N = total_timesteps / (n_envs * n_steps)).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.training_env.env_method("set_opponent_step", self.num_timesteps)
        # Log the current interpolated opponent mix so the schedule is visible in W&B.
        mixes = self.training_env.env_method("get_opponent_mix")
        mix = mixes[0] if mixes else None
        if mix:
            for name, weight in mix.items():
                self.logger.record(f"opponent/weight/{name}", float(weight))

    def _on_step(self) -> bool:
        return True


class SelfSnapshotCallback(BaseCallback):
    """Every `snapshot_every` env steps, save the current policy to `snapshot_path`
    so that `MixedWithSelf` opponent sampler can use it as the self-opponent.

    The save is atomic via temp file + rename so partial files never get loaded.
    """

    def __init__(self, snapshot_path: Path, snapshot_every: int, verbose: int = 0):
        super().__init__(verbose)
        self.snapshot_path = Path(snapshot_path)
        self.snapshot_every = snapshot_every
        self._last_snapshot_step = 0

    def _save(self) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        self.model.save(str(tmp))
        tmp.replace(self.snapshot_path)
        if self.verbose:
            print(f"[SelfSnapshot] step={self.num_timesteps} -> {self.snapshot_path}")

    def _on_training_start(self) -> None:
        # Snapshot once at the start so the 'self' opponent slot is a real (weak)
        # copy the moment the schedule introduces it, rather than falling back.
        if not self.snapshot_path.exists():
            self._save()
            self._last_snapshot_step = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snapshot_step < self.snapshot_every:
            return True
        self._last_snapshot_step = self.num_timesteps
        self._save()
        return True
