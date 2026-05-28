"""Factorized (action-type + parameter) policy as a MaskablePPO drop-in.

Charlesworth's main architectural lever (settlers_of_catan_RL) was a multi-head
policy with action-type + per-type parameter heads. We don't change the env's
flat `Discrete(332)` action space or its action mask — we only change how the
332 logits are *computed* inside the policy:

    logit(flat_a) = type_logit[type(flat_a)] + param_logit[type(flat_a)][param(flat_a)]

Heads share parameters where the semantic param-space is shared (notably:
`BUILD_SETTLEMENT` and `BUILD_CITY` both index over the 54 nodes and share one
node head). Trivial types (ROLL, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING,
END_TURN) get only a type-logit (no param head). `MARITIME_TRADE` is always
masked off and contributes nothing.

The distribution is still `MaskableCategoricalDistribution` over the same 332
flat actions, so everything downstream (sampling, log_prob, entropy, action
masking, MiniLadderCallback, RLPlayer round-trip) is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch as th
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType
from catanatron.models.player import Color


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------

_ALL_TYPES: list[ActionType] = list(ActionType)
_TYPE_TO_ID: dict[ActionType, int] = {t: i for i, t in enumerate(_ALL_TYPES)}

# Types whose param spaces are equivalent and share one head.
_SHARED_HEAD: dict[ActionType, str] = {
    ActionType.BUILD_SETTLEMENT: "node",
    ActionType.BUILD_CITY: "node",
}


def build_action_meta(player_colors: tuple[Color, ...], map_type: str = "BASE") -> dict[str, Any]:
    """Build the lookup tables a FactorizedActionNet needs.

    Returns a dict containing:
      * ``flat_to_type``    : int64 array (n_flat,) — ActionType id for each flat index
      * ``flat_to_param``   : int64 array (n_flat,) — param index *within* the head
      * ``type_to_head``    : dict[int, str | None] — head name for each type id
                              (`None` for trivial / masked types)
      * ``head_sizes``      : dict[str, int] — output dim per head
      * ``n_types``         : total # of ActionTypes (matches `len(ActionType)`)
      * ``n_flat``          : total # of flat actions
    """
    action_array = get_action_array(player_colors, map_type)
    n_flat = len(action_array)

    # Group flat indices by ActionType.
    by_type: dict[ActionType, list[tuple[int, Any]]] = defaultdict(list)
    for flat_idx, (atype, value) in enumerate(action_array):
        by_type[atype].append((flat_idx, value))

    flat_to_type = np.zeros(n_flat, dtype=np.int64)
    flat_to_param = np.zeros(n_flat, dtype=np.int64)
    type_to_head: dict[int, str | None] = {}
    head_sizes: dict[str, int] = {}

    # Mark every type id with its head decision.
    for tid in range(len(_ALL_TYPES)):
        type_to_head[tid] = None  # default: no param head

    for atype, entries in by_type.items():
        tid = _TYPE_TO_ID[atype]
        for flat_idx, _ in entries:
            flat_to_type[flat_idx] = tid

        if atype == ActionType.MARITIME_TRADE:
            # Always masked off; no head allocated.
            continue
        if len(entries) == 1:
            # Trivial type — only type_logit, no param.
            continue
        if atype in _SHARED_HEAD:
            head_name = _SHARED_HEAD[atype]
            # value IS the node_id (int); use it directly so the head
            # is genuinely shared between settlement and city.
            for flat_idx, value in entries:
                flat_to_param[flat_idx] = int(value)
            new_size = max(int(v) for _, v in entries) + 1
            head_sizes[head_name] = max(head_sizes.get(head_name, 0), new_size)
            type_to_head[tid] = head_name
        else:
            # Unshared head — param_idx is position in this type's sorted list
            # (action_array is already sorted by str(action), so position is stable).
            head_name = f"head_{atype.name}"
            for i, (flat_idx, _) in enumerate(entries):
                flat_to_param[flat_idx] = i
            head_sizes[head_name] = len(entries)
            type_to_head[tid] = head_name

    return {
        "flat_to_type": flat_to_type,
        "flat_to_param": flat_to_param,
        "type_to_head": type_to_head,
        "head_sizes": head_sizes,
        "n_types": len(_ALL_TYPES),
        "n_flat": n_flat,
    }


# ---------------------------------------------------------------------------
# FactorizedActionNet
# ---------------------------------------------------------------------------

class FactorizedActionNet(nn.Module):
    """Replacement for the default Linear(features → n_flat) action head."""

    def __init__(self, features_dim: int, meta: dict[str, Any]):
        super().__init__()
        self.meta = meta
        self.n_flat = meta["n_flat"]
        self.type_head = nn.Linear(features_dim, meta["n_types"])
        self.param_heads = nn.ModuleDict({
            name: nn.Linear(features_dim, size)
            for name, size in meta["head_sizes"].items()
        })
        # Lookup tensors live as buffers so .to(device) works automatically.
        self.register_buffer(
            "flat_to_type", th.from_numpy(meta["flat_to_type"]).long()
        )
        self.register_buffer(
            "flat_to_param", th.from_numpy(meta["flat_to_param"]).long()
        )
        # Per-head: which flat indices it contributes to and which param indices
        # in its own output correspond. Stored as buffers for fast gather.
        self._head_names: list[str] = list(meta["head_sizes"].keys())
        for name in self._head_names:
            tids_for_head = [
                tid for tid, hn in meta["type_to_head"].items() if hn == name
            ]
            flat_idxs: list[int] = []
            param_idxs: list[int] = []
            for f in range(self.n_flat):
                if int(meta["flat_to_type"][f]) in tids_for_head:
                    flat_idxs.append(f)
                    param_idxs.append(int(meta["flat_to_param"][f]))
            self.register_buffer(
                f"_h_{name}_flat", th.tensor(flat_idxs, dtype=th.long)
            )
            self.register_buffer(
                f"_h_{name}_param", th.tensor(param_idxs, dtype=th.long)
            )

    def forward(self, features: th.Tensor) -> th.Tensor:
        """Return (batch, n_flat) action logits."""
        type_logits = self.type_head(features)              # (B, n_types)
        flat_logits = type_logits.index_select(1, self.flat_to_type)  # (B, n_flat)
        # Add per-type parameter contributions.
        for name in self._head_names:
            head = self.param_heads[name]
            param_logits = head(features)                   # (B, head_size)
            flat_idxs: th.Tensor = getattr(self, f"_h_{name}_flat")
            param_idxs: th.Tensor = getattr(self, f"_h_{name}_param")
            chunk = param_logits.index_select(1, param_idxs)  # (B, |flat_idxs|)
            flat_logits = flat_logits.index_add(1, flat_idxs, chunk)
        return flat_logits


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class FactorizedMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Same as MaskableActorCriticPolicy, but with a factorized action head.

    Constructor extras:
      - ``player_colors``: tuple of Catanatron Color values (or color names) used
        to build the action-array layout. Defaults to (BLUE, RED) — the 1v1 setup.
      - ``map_type``: BASE / TOURNAMENT / MINI. Defaults to BASE.

    The meta dict is rebuilt at policy construction time (not stored on disk),
    so checkpoints only need to carry ``player_colors`` + ``map_type`` to
    reconstruct the policy.
    """

    def __init__(
        self,
        *args,
        player_colors: tuple = (Color.BLUE, Color.RED),
        map_type: str = "BASE",
        **kwargs,
    ):
        self._player_colors_raw = player_colors
        self._map_type = map_type
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Coerce strings to Color enums for serialization-tolerance.
        cs = tuple(
            Color[c] if isinstance(c, str) else c
            for c in self._player_colors_raw
        )
        meta = build_action_meta(cs, self._map_type)
        # latent_dim_pi is the size of the policy MLP-extractor output that the
        # default action_net consumed; we replace action_net with our factorized one.
        self.action_net = FactorizedActionNet(
            self.mlp_extractor.latent_dim_pi, meta
        )
        # Re-init optimizer so it picks up the new action_net params.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
