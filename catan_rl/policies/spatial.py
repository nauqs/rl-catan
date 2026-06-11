"""Spatial pointer policy: action logits for board locations read the *local*
board features at those locations.

Motivation
----------
`BoardCNN`/`BoardCNNAttn` pool the `(C, 21, 11)` board into one global vector,
so the action head must rank "build at node 42 vs node 17" with no idea what is
*at* node 42. ~237/332 actions are location choices. This policy fixes that: it
keeps the full conv feature map (via `HexEncoder`) and, for each location
action, gathers the feature column at that location's grid cell and scores it.

    logit(a) = type_bias[type(a)] + score(a)

where `score` for a node/edge/tile action is a small MLP over
`[ f_loc(a) ; g ]` (local features + global summary). SETTLEMENT and CITY use
*separate* scorers (a node can be great for a city but mediocre for a fresh
settlement — the factorized policy wrongly forced them to share one logit).
Non-spatial actions (maritime / year-of-plenty / monopoly / discard) use a
`Linear(d_g -> size)` head; trivial actions get only a type bias.

The distribution is still `MaskableCategoricalDistribution` over the same 332
flat actions, so masking / sampling / log-prob / entropy and everything
downstream (MaskablePPO, callbacks, `RLPlayer.predict()`) are unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from catanatron.gym.board_tensor_features import (
    get_node_and_edge_maps,
    get_tile_coordinate_map,
)
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType
from catanatron.models.player import Color


_ALL_TYPES: list[ActionType] = list(ActionType)
_TYPE_TO_ID: dict[ActionType, int] = {t: i for i, t in enumerate(_ALL_TYPES)}

# Non-spatial parameterized action types -> global head name.
_GLOBAL_TYPES: dict[ActionType, str] = {
    ActionType.MARITIME_TRADE: "maritime",
    ActionType.PLAY_YEAR_OF_PLENTY: "year_of_plenty",
    ActionType.PLAY_MONOPOLY: "monopoly",
    ActionType.DISCARD_RESOURCE: "discard",
}


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------

def build_spatial_action_meta(
    player_colors: tuple[Color, ...],
    map_type: str = "BASE",
    enable_maritime: bool = True,
) -> dict[str, Any]:
    """Precompute, for every flat action, how its logit is produced.

    Returns a dict with:
      * ``n_flat`` / ``n_types``
      * ``flat_to_type``  : int64 (n_flat,) — ActionType id per flat index
      * ``spatial``       : dict[kind -> list]; entries are
                            ``(flat_idx, i, j)`` for node_settle/node_city/edge
                            and ``(flat_idx, i, j, victim_id)`` for tile
      * ``globals``       : dict[head_name -> {"size", "entries":[(flat_idx, param_idx)]}]
      * ``n_victims``     : embedding size for the robber victim bias
      * ``enable_maritime``
    """
    action_array = get_action_array(player_colors, map_type)
    n_flat = len(action_array)
    node_map, edge_map = get_node_and_edge_maps()
    tile_map = get_tile_coordinate_map()

    flat_to_type = np.zeros(n_flat, dtype=np.int64)
    spatial: dict[str, list] = {
        "node_settle": [], "node_city": [], "edge": [], "tile": []
    }
    globals_: dict[str, dict] = {}
    global_counter: dict[str, int] = {}

    # Victim vocabulary (deterministic: None first, then by color name).
    victims = []
    for atype, value in action_array:
        if atype == ActionType.MOVE_ROBBER and value[1] not in victims:
            victims.append(value[1])
    victims_sorted = sorted(victims, key=lambda v: (v is not None, str(v)))
    victim_to_id = {v: i for i, v in enumerate(victims_sorted)}

    for flat_idx, (atype, value) in enumerate(action_array):
        flat_to_type[flat_idx] = _TYPE_TO_ID[atype]

        if atype == ActionType.BUILD_SETTLEMENT:
            i, j = node_map[int(value)]
            spatial["node_settle"].append((flat_idx, int(i), int(j)))
        elif atype == ActionType.BUILD_CITY:
            i, j = node_map[int(value)]
            spatial["node_city"].append((flat_idx, int(i), int(j)))
        elif atype == ActionType.BUILD_ROAD:
            i, j = edge_map[value]
            spatial["edge"].append((flat_idx, int(i), int(j)))
        elif atype == ActionType.MOVE_ROBBER:
            cube, victim = value
            i, j = tile_map[cube]
            spatial["tile"].append((flat_idx, int(i), int(j), victim_to_id[victim]))
        elif atype in _GLOBAL_TYPES:
            if atype == ActionType.MARITIME_TRADE and not enable_maritime:
                continue  # masked off -> type bias only, no head allocated
            name = _GLOBAL_TYPES[atype]
            param_idx = global_counter.get(name, 0)
            global_counter[name] = param_idx + 1
            globals_.setdefault(name, {"entries": []})["entries"].append(
                (flat_idx, param_idx)
            )
        # else: trivial type (ROLL, END_TURN, BUY_DEVELOPMENT_CARD,
        # PLAY_KNIGHT_CARD, PLAY_ROAD_BUILDING) -> type bias only.

    for spec in globals_.values():
        spec["size"] = len(spec["entries"])

    return {
        "n_flat": n_flat,
        "n_types": len(_ALL_TYPES),
        "flat_to_type": flat_to_type,
        "spatial": spatial,
        "globals": globals_,
        "n_victims": max(1, len(victims_sorted)),
        "enable_maritime": enable_maritime,
    }


# ---------------------------------------------------------------------------
# SpatialActionHead
# ---------------------------------------------------------------------------

class SpatialActionHead(nn.Module):
    """Produces (B, n_flat) logits from a conv feature map + global vector."""

    _SPATIAL = [
        ("node_settle", "settle_mlp"),
        ("node_city", "city_mlp"),
        ("edge", "edge_mlp"),
        ("tile", "tile_mlp"),
    ]

    def __init__(
        self,
        meta: dict[str, Any],
        *,
        map_shape: tuple[int, int, int],
        d_g: int,
        hidden: int = 128,
    ):
        super().__init__()
        c, _h, w = map_shape
        self.n_flat = int(meta["n_flat"])
        self._w = int(w)

        self.type_bias = nn.Parameter(th.zeros(meta["n_types"]))
        self.register_buffer(
            "flat_to_type", th.from_numpy(meta["flat_to_type"]).long()
        )

        self.settle_mlp = self._mk_mlp(c + d_g, hidden)
        self.city_mlp = self._mk_mlp(c + d_g, hidden)
        self.edge_mlp = self._mk_mlp(c + d_g, hidden)
        self.tile_mlp = self._mk_mlp(c + d_g, hidden)
        self.victim_emb = nn.Embedding(meta["n_victims"], 1)

        for kind, _ in self._SPATIAL:
            entries = meta["spatial"][kind]
            flat = [e[0] for e in entries]
            cell = [e[1] * self._w + e[2] for e in entries]  # row-major i*W + j
            self.register_buffer(f"_{kind}_flat", th.tensor(flat, dtype=th.long))
            self.register_buffer(f"_{kind}_cell", th.tensor(cell, dtype=th.long))
            if kind == "tile":
                vic = [e[3] for e in entries]
                self.register_buffer("_tile_victim", th.tensor(vic, dtype=th.long))

        self.global_heads = nn.ModuleDict()
        self._global_names: list[str] = []
        for name, spec in meta["globals"].items():
            self.global_heads[name] = nn.Linear(d_g, spec["size"])
            flat = [e[0] for e in spec["entries"]]
            param = [e[1] for e in spec["entries"]]
            self.register_buffer(f"_g_{name}_flat", th.tensor(flat, dtype=th.long))
            self.register_buffer(f"_g_{name}_param", th.tensor(param, dtype=th.long))
            self._global_names.append(name)

    @staticmethod
    def _mk_mlp(in_dim: int, hidden: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_map: th.Tensor, g: th.Tensor) -> th.Tensor:
        b = g.shape[0]
        c = feat_map.shape[1]
        map_flat = feat_map.reshape(b, c, -1)  # (B, C, H*W)

        out = self.type_bias.index_select(0, self.flat_to_type)  # (n_flat,)
        out = out.unsqueeze(0).repeat(b, 1)                      # (B, n_flat)

        for kind, mlp_name in self._SPATIAL:
            flat_idx: th.Tensor = getattr(self, f"_{kind}_flat")
            if flat_idx.numel() == 0:
                continue
            cell: th.Tensor = getattr(self, f"_{kind}_cell")
            f_loc = map_flat.index_select(2, cell).permute(0, 2, 1)  # (B, K, C)
            k = f_loc.shape[1]
            gg = g.unsqueeze(1).expand(b, k, g.shape[1])
            scores = getattr(self, mlp_name)(th.cat([f_loc, gg], dim=2)).squeeze(-1)
            if kind == "tile":
                vbias = self.victim_emb(self._tile_victim).squeeze(-1)  # (K,)
                scores = scores + vbias.unsqueeze(0)
            out = out.index_add(1, flat_idx, scores)

        for name in self._global_names:
            head_out = self.global_heads[name](g)               # (B, size)
            flat_idx = getattr(self, f"_g_{name}_flat")
            param = getattr(self, f"_g_{name}_param")
            out = out.index_add(1, flat_idx, head_out.index_select(1, param))

        return out


# ---------------------------------------------------------------------------
# Attention value head (optional): the critic attends over the spatial map
# instead of reading only the pooled global vector. The policy head has always
# had location-level access; this gives the critic the same, fixing the
# structurally-coarse V(s) that miscalibrates off-distribution.
# ---------------------------------------------------------------------------

class AttnValueHead(nn.Module):
    def __init__(self, c_map: int, d_g: int, n_heads: int = 4, hidden: int = 256):
        super().__init__()
        self.query = nn.Parameter(th.randn(1, 1, c_map) * 0.02)
        self.attn = nn.MultiheadAttention(c_map, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(c_map + d_g, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_map: th.Tensor, g: th.Tensor) -> th.Tensor:
        b = feat_map.shape[0]
        seq = feat_map.flatten(2).transpose(1, 2)          # (B, H*W, C)
        pooled, _ = self.attn(self.query.expand(b, -1, -1), seq, seq)
        return self.mlp(th.cat([pooled.squeeze(1), g], dim=1))


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class SpatialMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy whose action/value heads read the spatial map.

    Requires the feature extractor to be ``HexEncoder`` (returns a flat
    ``concat[ flatten(map), g ]`` tensor that this policy splits back). The
    default ``mlp_extractor`` / ``action_net`` path is bypassed.

    Constructor extras (mirror ``FactorizedMaskableActorCriticPolicy``):
      - ``player_colors`` / ``map_type`` — used to build the action layout.
      - ``enable_maritime`` — whether the maritime head exists (must match the
        env's ``NoTradeWrapper(strip_maritime=...)`` setting).
    """

    def __init__(
        self,
        *args,
        player_colors: tuple = (Color.BLUE, Color.RED),
        map_type: str = "BASE",
        enable_maritime: bool = True,
        value_head: str = "mlp",   # "mlp" (legacy, reads g only) | "attn" (attends over map)
        **kwargs,
    ):
        self._player_colors_raw = player_colors
        self._map_type = map_type
        self._enable_maritime = enable_maritime
        self._value_head_kind = value_head
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        enc = self.features_extractor
        self._map_shape = tuple(int(s) for s in enc.map_shape)
        self._d_g = int(enc.d_g)

        cs = tuple(
            Color[c] if isinstance(c, str) else c for c in self._player_colors_raw
        )
        meta = build_spatial_action_meta(cs, self._map_type, self._enable_maritime)
        self.action_head = SpatialActionHead(
            meta, map_shape=self._map_shape, d_g=self._d_g
        )
        # Value head: legacy MLP reads g only; "attn" attends over the feature map.
        if self._value_head_kind == "attn":
            self.value_net = AttnValueHead(self._map_shape[0], self._d_g)
        else:
            self.value_net = nn.Sequential(
                nn.Linear(self._d_g, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )
        # Drop the bypassed default heads so they don't bloat the checkpoint.
        self.action_net = nn.Identity()
        self.mlp_extractor = nn.Identity()

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ---- feature split -----------------------------------------------------
    def _split(self, feats: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        b = feats.shape[0]
        c, h, w = self._map_shape
        n_map = c * h * w
        feat_map = feats[:, :n_map].view(b, c, h, w)
        g = feats[:, n_map:]
        return feat_map, g

    def _value(self, feat_map: th.Tensor, g: th.Tensor) -> th.Tensor:
        if self._value_head_kind == "attn":
            return self.value_net(feat_map, g)
        return self.value_net(g)

    def _logits_and_value(self, obs) -> tuple[th.Tensor, th.Tensor]:
        feat_map, g = self._split(self.extract_features(obs))
        return self.action_head(feat_map, g), self._value(feat_map, g)

    # ---- overrides (keep masking / sampling / log-prob / entropy intact) ----
    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self._logits_and_value(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._logits_and_value(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs, action_masks=None):
        feat_map, g = self._split(self.extract_features(obs))
        dist = self.action_dist.proba_distribution(
            action_logits=self.action_head(feat_map, g)
        )
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def predict_values(self, obs):
        feat_map, g = self._split(self.extract_features(obs))
        return self._value(feat_map, g)
