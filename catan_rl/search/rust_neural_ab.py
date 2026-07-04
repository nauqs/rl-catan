"""AlphaBeta search guided by a Rust-engine-trained net, inside Catanatron.

The Phase-7 recipe (policy-top-k pruned expectiminimax — our first near-parity
with AB2) re-plumbed for Phase-10 nets via the cross-engine adapter:

  - PRUNING: at every search node, rank the mover's legal actions with the
    net's policy logits (seat-relative obs from the mover's perspective) and
    keep the top-k. This is the load-bearing part (AlphaZero's lesson: search
    the policy's moves, evaluate to choose among them).
  - LEAF EVAL: either Catanatron's handcrafted base_fn (leaf_value="basefn" —
    lets pruning savings buy DEPTH with the same evaluator AB2 uses) or the
    net's value head (leaf_value="net" — tests the learned evaluator; beware
    it was trained on 4p first-to-7 returns).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from catanatron.players.minimax import AlphaBetaPlayer

from catan_rl.eval.adapter_maps import build_maps
from catan_rl.eval.rust_net_player import (
    NUM_ACTIONS, catanatron_state_to_obs, encode_action, load_rust_net,
)


class RustNetAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self, color, model_path: str | Path, *, depth: int = 2,
                 policy_topk: int = 3, leaf_value: str = "basefn",
                 vps_to_win: int = 7, device: str = "cpu",
                 arch: str = "hexformer", evaluator_path: str | Path | None = None,
                 **net_kw):
        super().__init__(color, depth=depth, prunning=False)
        assert leaf_value in ("basefn", "net", "outcome")
        self.maps = build_maps()
        self.net = load_rust_net(model_path, device=device, arch=arch, **net_kw)
        self.device = device
        self.policy_topk = int(policy_topk)
        self.vps_to_win = vps_to_win
        self.leaf_value = leaf_value
        # outcome: dedicated state->P(win) regressor trained on final results
        # — an alternative to learned value heads as leaves.
        self.evaluator = None
        if leaf_value == "outcome":
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                                   / "scripts" / "rust_train"))
            from train_evaluator import OutcomeEvaluator
            ck = torch.load(str(evaluator_path), map_location=device,
                            weights_only=False)
            self.evaluator = OutcomeEvaluator()
            self.evaluator.load_state_dict(ck["model_state"])
            self.evaluator.to(device).eval()
        # basefn: leave use_value_function False -> catanatron uses base_fn.
        self.use_value_function = leaf_value in ("net", "outcome")

    def _logits(self, game, color, ids):
        obs = catanatron_state_to_obs(game, color, self.maps, self.vps_to_win)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ids] = True
        with torch.no_grad():
            o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            logits, value = self.net(o, m)
        return logits[0], float(value.reshape(-1)[0])

    def get_actions(self, game):
        actions = list(super().get_actions(game))
        if self.policy_topk <= 0 or len(actions) <= self.policy_topk:
            return actions
        mover = game.state.current_color()
        pairs = [(a, encode_action(a, self.maps)) for a in actions]
        ids = [i for _, i in pairs if i is not None]
        if not ids:
            return actions[: self.policy_topk]
        logits, _ = self._logits(game, mover, ids)
        scored = sorted(
            pairs, key=lambda p: float(logits[p[1]]) if p[1] is not None else -1e30,
            reverse=True,
        )
        return [a for a, _ in scored[: self.policy_topk]]

    def value_function(self, game, p0_color):
        # only called when use_value_function=True ("net" or "outcome")
        if self.evaluator is not None:
            obs = catanatron_state_to_obs(game, p0_color, self.maps, self.vps_to_win)
            with torch.no_grad():
                o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                return float(self.evaluator(o)[0])
        _, v = self._logits(game, p0_color, [0])
        return v
