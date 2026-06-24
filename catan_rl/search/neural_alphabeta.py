"""AlphaBeta search with our trained value net as the leaf evaluator.

Catanatron's `AlphaBetaPlayer` is expectiminimax to `depth` plies; at leaves it
scores positions with the handcrafted `base_fn` (the SAME function
`ValueFunctionPlayer` uses greedily). So the ladder is one value fn at increasing
search depth: value=base_fn@d0, AB1=base_fn@d1, AB2=base_fn@d2.

This subclass swaps the leaf eval for our LEARNED value net, via Catanatron's
built-in hook: `alphabeta` calls `get_value_fn(..., self.value_function if
self.use_value_function else None)`, and `get_value_fn` returns our function
directly when provided. So setting `use_value_function=True` + overriding
`value_function(game, p0_color)` is all it takes.

Purpose: test whether a *learned* value function beats the *handcrafted* one inside
identical search — and serve as iteration-0 of the AlphaBeta-ExIt loop (self-play
with this player -> retrain the value net on outcomes -> iterate, so the value fn
climbs above the handcrafted baseline).

Note on perspective/scale: `alphabeta` always calls `value_fn(game, self.color)`,
so we evaluate from our own color's perspective (higher = better for us), exactly
like `base_fn`. The minimax/expectimax structure handles whose turn it is, and
alphabeta is scale-invariant, so the net's raw return estimate needs no rescaling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th

from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.gym.envs.action_space import to_action_space

from catan_rl.players.rl_player import RLPlayer


class NeuralAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(
        self,
        color: Color,
        model_path: str | Path,
        *,
        depth: int = 1,
        device: str = "cpu",          # CPU: single-position leaf evals, MPS per-call overhead hurts
        representation: str = "mixed",
        enable_maritime_trade: bool = True,
        prunning: bool = False,        # match stock AB (False) for a clean value-fn-swap comparison
        policy_topk: int = 0,          # >0: restrict the search to the policy's top-k moves
        value_model_path: str | Path | None = None,  # optional: separate net for LEAF VALUE
    ):
        super().__init__(color, depth=depth, prunning=prunning)
        self.use_value_function = True
        self.policy_topk = int(policy_topk)
        # `model_path` provides the POLICY (for top-k pruning). `value_model_path`, if
        # given, provides a DIFFERENT net for the LEAF VALUE (so we can hold a strong
        # policy fixed and A/B different value evaluators). Obs-building is model-agnostic.
        self._rl = RLPlayer(
            color, model_path, device=device,
            representation=representation, enable_maritime_trade=enable_maritime_trade,
        )
        self._policy = self._rl._model.policy
        self._policy.set_training_mode(False)
        if value_model_path is not None:
            self._rl_v = RLPlayer(
                color, value_model_path, device=device,
                representation=representation, enable_maritime_trade=enable_maritime_trade,
            )
            self._value_policy = self._rl_v._model.policy
        else:
            self._value_policy = self._policy
        self._value_policy.set_training_mode(False)

    def get_actions(self, game):
        """Restrict the branching to the policy's top-k legal moves.

        Iteration-0 (value net as the ONLY leaf eval, no policy) collapsed: a minimax
        search driven by a policy-conditioned value net hunts for off-distribution
        states the net over-rates. BC's strength is its POLICY (imitates value's good
        moves), so we keep the search ON those moves and use the value only to choose
        among them via lookahead -- exactly why AlphaZero's PUCT uses the policy prior.
        Prunes for whichever color is to move (self-play assumption: our net models
        both sides; obs/policy are color-relative)."""
        actions = list(super().get_actions(game))
        if self.policy_topk <= 0 or len(actions) <= self.policy_topk:
            return actions
        pcs = tuple(p.color for p in game.state.players)
        color = game.state.current_color()
        obs = self._rl._build_obs(game, color)
        mask = self._rl._build_mask(pcs, actions)
        tensor, _ = self._policy.obs_to_tensor(obs)
        with th.no_grad():
            dist = self._policy.get_distribution(tensor, action_masks=mask[None, :])
            probs = dist.distribution.probs.reshape(-1).cpu().numpy()
        scored = sorted(
            actions, key=lambda a: probs[to_action_space(a, pcs, self._rl._map_type)],
            reverse=True,
        )
        return scored[: self.policy_topk]

    def value_function(self, game, p0_color):
        obs = self._rl._build_obs(game, p0_color)
        tensor, _ = self._value_policy.obs_to_tensor(obs)
        with th.no_grad():
            v = self._value_policy.predict_values(tensor)
        return float(v.reshape(-1)[0].item())
