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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.opponents = opponents
        self.n_games = n_games
        self.save_dir = Path(save_dir)
        self.representation = representation
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
        from catan_rl.players.rl_player import RLPlayer

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "current.zip"
            self.model.save(str(ckpt))

            def factory(color):
                return RLPlayer(
                    color, ckpt, representation=self.representation, deterministic=True
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

        if self.verbose:
            print(f"[MiniLadder] step={self.num_timesteps} best_random_wr={self._best_random_wr:.2f}")
