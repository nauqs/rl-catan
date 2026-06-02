"""UCB1 root-search player (Phase 4 lever B/D).

Inference-time, single-step UCB1 expansion at the root with short rollouts
bootstrapped by the trained value net at the leaf. Wraps a trained
``MaskablePPO`` checkpoint and reuses ``RLPlayer``'s observation / action-mask
machinery, so it plugs into the ladder eval exactly like ``RLPlayer``.

Rollout policy (``rollout_policy``):
  * ``"policy"`` (default) — the trained net drives the rollout, sampling
    actions from its (masked) distribution. With ``policy_opponent=True`` it
    also plays the opponents from *their* perspective, i.e. self-play planning
    in the style of Charlesworth's forward-search agent. ~2-4x the cost of
    random rollouts but a far stronger default policy.
  * ``"random"`` — uniform-random rollout (cheapest; weak signal).

For each decision the agent faces:

  1. Enumerate the legal root actions (maritime stripped if the model was
     trained with it disabled, matching ``RLPlayer``).
  2. Allocate ``n_sims`` simulations across them with UCB1. Each simulation
     copies the game (``game.copy()`` ~ 10us), applies the candidate root
     action, plays ``rollout_depth`` cheap plies (``decide_fn`` overrides every
     player's choice, so there is no recursion back into search), then scores
     the leaf: real terminal win/loss, else the value net.
  3. Pick the root action with the best mean value.

Leaf values are normalised by ``value_scale`` (default 500, the terminal
win reward) so Q lands in roughly [-1, 1] and the UCB exploration constant
behaves sensibly. Perfect-information rollouts (the copied game carries true
opponent hands) — acceptable for a baseline search lever; hidden-hand
determinization is a future refinement.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import torch as th

from catanatron.game import TURNS_LIMIT
from catanatron.models.player import Color

from catan_rl.env.no_trade import TRADE_ACTION_TYPES
from catan_rl.players.rl_player import RLPlayer


class SearchRLPlayer(RLPlayer):
    def __init__(
        self,
        color: Color,
        model_path: str | Path,
        *,
        n_sims: int = 64,
        rollout_depth: int = 8,
        c_ucb: float = 1.4,
        rollout_policy: str = "policy",   # "policy" (net, self-play) or "random" (cheap)
        policy_opponent: bool = True,     # net also drives opponents (Charlesworth self-play)
        value_scale: float = 500.0,
        win_value: float = 500.0,
        loss_value: float = -500.0,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        **rl_kwargs,
    ):
        super().__init__(color, model_path, deterministic=True, seed=seed, **rl_kwargs)
        self.n_sims = int(n_sims)
        self.rollout_depth = int(rollout_depth)
        self.c_ucb = float(c_ucb)
        self.rollout_policy = rollout_policy
        self.policy_opponent = bool(policy_opponent)
        self.value_scale = float(value_scale)
        self.win_value = float(win_value)
        self.loss_value = float(loss_value)
        # temperature > 0: sample the played move from the visit distribution
        # N(a)^(1/T) (Expert-Iteration exploration); 0 = greedy argmax(mean Q).
        self.temperature = float(temperature)
        self._sr = random.Random(seed)
        # ExIt hook: last root search result, set every multi-action decide().
        # last_actions: list of catanatron Action; last_visits: parallel counts.
        self.last_actions = None
        self.last_visits = None

    # --- leaf evaluation ----------------------------------------------------
    def _value_net(self, game) -> float:
        obs = self._build_obs(game)
        tensor, _ = self._model.policy.obs_to_tensor(obs)
        with th.no_grad():
            v = self._model.policy.predict_values(tensor)
        return float(v.flatten()[0])

    def _leaf_value(self, sim) -> float:
        """Normalised value of a rollout leaf from self.color's perspective."""
        winner = sim.winning_color()
        if winner is not None:
            raw = self.win_value if winner == self.color else self.loss_value
        else:
            raw = self._value_net(sim)
        return raw / self.value_scale

    # --- rollout policy -----------------------------------------------------
    def _rollout_decide(self, player, game, actions):
        if len(actions) == 1:
            return actions[0]
        use_net = self.rollout_policy == "policy" and (
            player.color == self.color or self.policy_opponent
        )
        if use_net:
            pcs = tuple(p.color for p in game.state.players)
            # Build the observation from the *acting* player's perspective so
            # the (perspective-relative) net can play either side (self-play).
            obs = self._build_obs(game, color=player.color)
            mask = self._build_mask(pcs, actions)
            if mask.any():
                from catanatron.gym.envs.action_space import from_action_space
                # Sampled (not greedy) so repeated sims explore different lines.
                a_int, _ = self._model.predict(obs, action_masks=mask, deterministic=False)
                decoded = from_action_space(int(a_int), player.color, pcs, self._map_type)
                if decoded in actions:
                    return decoded
                for a in actions:
                    if a.action_type == decoded.action_type and a.value == decoded.value:
                        return a
        return self._sr.choice(list(actions))

    def _simulate(self, game, root_action) -> float:
        sim = game.copy()
        sim.execute(root_action)
        for _ in range(self.rollout_depth):
            if sim.winning_color() is not None or sim.state.num_turns >= TURNS_LIMIT:
                break
            sim.play_tick(decide_fn=self._rollout_decide)
        return self._leaf_value(sim)

    # --- root search --------------------------------------------------------
    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if not self._enable_maritime_trade:
            filtered = [a for a in actions if a.action_type not in TRADE_ACTION_TYPES]
            if filtered:
                actions = filtered
        if len(actions) == 1:
            self.last_actions = None
            self.last_visits = None
            return actions[0]

        n = len(actions)
        visits = [0] * n
        total_q = [0.0] * n
        total = 0

        # One mandatory sample per arm, then UCB1 for the remaining budget.
        budget = max(self.n_sims, n)
        for i in range(n):
            total_q[i] += self._simulate(game, actions[i])
            visits[i] += 1
            total += 1

        while total < budget:
            log_t = math.log(total + 1)
            best_u, best_i = -float("inf"), 0
            for i in range(n):
                q = total_q[i] / visits[i]
                u = q + self.c_ucb * math.sqrt(log_t / visits[i])
                if u > best_u:
                    best_u, best_i = u, i
            total_q[best_i] += self._simulate(game, actions[best_i])
            visits[best_i] += 1
            total += 1

        # Expose the root visit distribution for ExIt distillation.
        self.last_actions = actions
        self.last_visits = list(visits)

        if self.temperature > 0:
            # Sample from N(a)^(1/T) (AlphaZero-style exploration during gen).
            w = [v ** (1.0 / self.temperature) for v in visits]
            s = sum(w)
            if s > 0:
                r, acc = self._sr.random() * s, 0.0
                for i, wi in enumerate(w):
                    acc += wi
                    if r <= acc:
                        return actions[i]
        means = [total_q[i] / visits[i] for i in range(n)]
        return actions[max(range(n), key=means.__getitem__)]
