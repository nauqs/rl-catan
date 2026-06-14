"""MaskablePPO with a KL-to-magnet anchor + MMD-style regularization.

Background (the project's central blocker). Three BC->PPO runs (v1 naked, v2
critic-warmup, v3 lean-reward) all collapsed the same way: PPO's policy gradient
drifts the policy off the BC demonstration manifold, the critic can't track it
there, advantages go bad, and competence bleeds to zero. A *fixed* KL-to-BC
anchor (the original use of this class) STOPS the collapse but CAPS at BC --
exceeding BC means leaving the very manifold the fixed anchor defends.

The fix (from "Reevaluating Policy Gradient Methods for IIG", arXiv 2502.08938,
and Magnetic Mirror Descent / R-NaD): make the magnet MOVE. MMD's update is
    argmax_pi  E[q]  -  alpha*KL(pi||rho)  -  (1/eta)*KL(pi||pi_prev)
i.e. a magnet term toward a reference rho *and* a proximal term toward the
previous policy. A magnet that tracks the policy (EMA, or periodic reset to the
current policy) keeps every step stable -- so no collapse -- yet does not pin
competence at any fixed level, so the policy can climb past BC.

This subclass supports:
  * magnet_mode "fixed"  -- KL(pi||pi_ref) with pi_ref frozen at the BC seed.
  * magnet_mode "ema"    -- pi_ref tracks pi by EMA each rollout (moving magnet).
  * magnet_mode "reset"  -- pi_ref <- copy(pi) every magnet_reset_steps steps.
  * mmd_kl_coef          -- the MMD proximal term, backward-KL(pi||pi_old).
  * ent_coef_final       -- linear entropy anneal (high-entropy IIG regime).

Everything except the extra terms / magnet bookkeeping is copied verbatim from
``sb3_contrib.ppo_mask.MaskablePPO.train`` (there is no hook to override).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class KLAnchoredMaskablePPO(MaskablePPO):
    """MaskablePPO + (moving) magnet KL penalty + MMD proximal term."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef: float = 0.0
        self.kl_target: float = 0.0          # >0 -> adaptive coef
        self.magnet_mode: str = "fixed"
        self.magnet_ema: float = 0.01
        self.magnet_reset_steps: int = 0
        self.mmd_kl_coef: float = 0.0
        self.ent_coef_start: float | None = None
        self.ent_coef_final: float | None = None
        self._total_timesteps_hint: int = 0
        self._ref_policy = None
        self._magnet_state_path: Path | None = None
        self._last_magnet_reset: int = 0

    def set_kl_anchor(
        self,
        ref_path: str,
        kl_coef: float,
        kl_target: float = 0.0,
        *,
        magnet_mode: str = "fixed",
        magnet_ema: float = 0.01,
        magnet_reset_steps: int = 0,
        mmd_kl_coef: float = 0.0,
        ent_coef_final: float | None = None,
        total_timesteps: int = 0,
        magnet_state_path: str | None = None,
    ) -> None:
        """Load the magnet reference from `ref_path` and configure the penalties.

        kl_target > 0 turns on the adaptive controller (kl_coef tracks the measured
        KL toward kl_target). magnet_mode in {fixed, ema, reset} controls whether
        the magnet moves. If `magnet_state_path` exists, the *moved* magnet is
        loaded from there instead of `ref_path` (resume continuity, so watchdog
        restarts don't yank the magnet back to the BC seed -> back to the BC cap).
        """
        self.kl_coef = float(kl_coef)
        self.kl_target = float(kl_target)
        self.magnet_mode = str(magnet_mode)
        self.magnet_ema = float(magnet_ema)
        self.magnet_reset_steps = int(magnet_reset_steps)
        self.mmd_kl_coef = float(mmd_kl_coef)
        self.ent_coef_start = float(self.ent_coef)
        self.ent_coef_final = None if ent_coef_final is None else float(ent_coef_final)
        self._total_timesteps_hint = int(total_timesteps)
        self._magnet_state_path = Path(magnet_state_path) if magnet_state_path else None

        # A reference policy is only needed for the magnet KL term (kl_coef>0) or a
        # moving magnet. With kl_coef==0 (e.g. PPO-from-scratch using only the MMD
        # proximal and/or entropy anneal) there is no BC seed to load -> ref stays None.
        if ref_path is not None:
            ref_model = MaskablePPO.load(ref_path, device=self.device)
            self._ref_policy = ref_model.policy
            # Continue a previously-moved magnet across resumes.
            if self._magnet_state_path is not None and self._magnet_state_path.exists():
                try:
                    state = th.load(self._magnet_state_path, map_location=self.device)
                    self._ref_policy.load_state_dict(state)
                    if self.verbose:
                        print(f"[kl-anchor] resumed moved magnet from {self._magnet_state_path}")
                except Exception as e:  # noqa: BLE001
                    print(f"[kl-anchor] could not load magnet state ({e}); using {ref_path}")
            self._ref_policy.set_training_mode(False)
            for p in self._ref_policy.parameters():
                p.requires_grad_(False)
        else:
            self._ref_policy = None
        if self.verbose:
            kl_mode = f"adaptive target={self.kl_target}" if self.kl_target > 0 else "fixed-coef"
            print(f"[kl-anchor] ref={ref_path} kl_coef={self.kl_coef} ({kl_mode}) "
                  f"magnet={self.magnet_mode} ema={self.magnet_ema} "
                  f"reset_steps={self.magnet_reset_steps} mmd_kl_coef={self.mmd_kl_coef} "
                  f"ent {self.ent_coef_start}->{self.ent_coef_final}")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _adapt_kl_coef(self, measured_kl: float) -> None:
        """RLHF-style proportional controller toward kl_target (clamped)."""
        if self.kl_target <= 0:
            return
        err = measured_kl / self.kl_target - 1.0
        err = max(-0.5, min(0.5, err))          # clamp per-step move
        self.kl_coef *= (1.0 + 0.1 * err)        # gentle multiplicative update
        self.kl_coef = max(1e-3, min(50.0, self.kl_coef))

    def _anneal_ent_coef(self) -> None:
        """Linear entropy anneal start -> final over the run (if configured)."""
        if self.ent_coef_final is None or self._total_timesteps_hint <= 0:
            return
        frac = min(1.0, max(0.0, self.num_timesteps / self._total_timesteps_hint))
        self.ent_coef = (1.0 - frac) * self.ent_coef_start + frac * self.ent_coef_final

    @th.no_grad()
    def _update_magnet(self) -> None:
        """Move the magnet toward the current policy (ema / reset modes)."""
        if self._ref_policy is None or self.magnet_mode == "fixed":
            return
        if self.magnet_mode == "ema":
            tau = self.magnet_ema
            for rp, pp in zip(self._ref_policy.parameters(), self.policy.parameters()):
                rp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)
            # buffers (e.g. running stats) tracked too, hard-copied
            for rb, pb in zip(self._ref_policy.buffers(), self.policy.buffers()):
                rb.data.copy_(pb.data)
        elif self.magnet_mode == "reset":
            if self.magnet_reset_steps > 0 and (
                self.num_timesteps - self._last_magnet_reset >= self.magnet_reset_steps
            ):
                self._ref_policy.load_state_dict(self.policy.state_dict())
                self._last_magnet_reset = self.num_timesteps
                if self.verbose:
                    print(f"[kl-anchor] magnet reset to current policy @ {self.num_timesteps}")
        if self._magnet_state_path is not None:
            try:
                th.save(self._ref_policy.state_dict(), self._magnet_state_path)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # train (verbatim MaskablePPO.train + magnet/MMD/ent-anneal terms)
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._anneal_ent_coef()
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_anchor_losses = []
        mmd_prox_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- magnet KL: kl_coef * KL(pi || pi_ref) -------------------------
                if self._ref_policy is not None and self.kl_coef > 0:
                    cur_dist = self.policy.get_distribution(
                        rollout_data.observations,
                        action_masks=rollout_data.action_masks,
                    ).distribution
                    with th.no_grad():
                        ref_dist = self._ref_policy.get_distribution(
                            rollout_data.observations,
                            action_masks=rollout_data.action_masks,
                        ).distribution
                    kl_anchor = th.distributions.kl_divergence(cur_dist, ref_dist).mean()
                    kl_anchor_losses.append(kl_anchor.item())
                    loss = loss + self.kl_coef * kl_anchor

                # --- MMD proximal: mmd_kl_coef * KL(pi_new || pi_old) --------------
                # k3 estimator, samples drawn from pi_old (the rollout policy).
                if self.mmd_kl_coef > 0:
                    log_ratio_p = log_prob - rollout_data.old_log_prob
                    mmd_prox = (ratio * log_ratio_p - (ratio - 1.0)).mean()
                    mmd_prox_losses.append(mmd_prox.item())
                    loss = loss + self.mmd_kl_coef * mmd_prox
                # -------------------------------------------------------------------

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Move the magnet toward the (now-updated) policy.
        self._update_magnet()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/ent_coef", self.ent_coef)
        if mmd_prox_losses:
            self.logger.record("train/mmd_proximal", float(np.mean(mmd_prox_losses)))
        if kl_anchor_losses:
            mean_kl = float(np.mean(kl_anchor_losses))
            self.logger.record("train/kl_anchor", mean_kl)
            self._adapt_kl_coef(mean_kl)
            self.logger.record("train/kl_coef", self.kl_coef)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
