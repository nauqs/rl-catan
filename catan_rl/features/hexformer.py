"""Tile Transformer trunk (HexFormer) — adjacency-aware drop-in for HexEncoder.

Motivation
----------
``BoardCNN``/``HexEncoder`` run 3x3 convs over Catanatron's 21x11 *raster*
projection of the hex board, so a kernel mixes raster-neighbour cells that only
loosely match true hex adjacency. ``HexFormer`` instead tokenises the board into
the 19 tiles + 54 nodes and runs graph-masked self-attention along the *real*
board adjacency (tile<->node incidence, node<->node edges, tile<->tile), then
scatters the adjacency-refined node/tile features back onto the raster grid.

Drop-in contract
----------------
Exposes ``map_shape`` (C, H, W) and ``d_g`` and returns the flat layout
``[ feat_map.flatten(1) , g ]`` — exactly what ``SpatialMaskableActorCriticPolicy``
expects, so the spatial pointer head / maritime head / value head are reused
unchanged. The *only* thing that differs from the Phase 3 conv run is this trunk.
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from catan_rl.features.hex_graph import N_TOKENS, build_hex_graph

# Final feature-map channels gathered per cell by the pointer head.
MAP_CHANNELS = 64


class HexFormer(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        d_g: int = 256,
        conv_channels: tuple[int, int] = (32, MAP_CHANNELS),
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        numeric_hidden: int = 256,
    ):
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]
        in_ch, height, width = (int(s) for s in board_space.shape)
        c1, c_map = conv_channels
        assert c_map == MAP_CHANNELS, "last conv channel must equal MAP_CHANNELS"

        n_map = c_map * height * width
        super().__init__(observation_space, features_dim=n_map + d_g)
        self.map_shape = (c_map, height, width)
        self.d_g = d_g
        self._n_map = n_map
        self._c_map = c_map

        # Local conv stem (keeps the raster spatial map for edge/road cells).
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c_map, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Token embedding: gathered conv column + type + id.
        self.tok_in = nn.Linear(c_map, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0=tile, 1=node
        self.id_emb = nn.Embedding(N_TOKENS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.tok_out = nn.Linear(d_model, c_map)

        n_numeric = int(numeric_space.shape[0])
        self.numeric_net = nn.Sequential(
            nn.LayerNorm(n_numeric),
            nn.Linear(n_numeric, numeric_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(numeric_hidden, d_g - c_map),
            nn.ReLU(inplace=True),
        )

        # Static graph buffers (template-fixed for BASE).
        g = build_hex_graph(height, width)
        self.register_buffer("_token_cells", th.from_numpy(g.token_cells).long())
        self.register_buffer("_token_type", th.from_numpy(g.token_type).long())
        self.register_buffer("_token_ids", th.arange(N_TOKENS).long())
        attn = th.where(
            th.from_numpy(g.adj_mask), 0.0, float("-inf")
        ).float()  # (T, T): 0 allowed, -inf blocked
        self.register_buffer("_attn_mask", attn)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        board = observations["board"].float()
        numeric = observations["numeric"].float()
        b = board.shape[0]
        c = self._c_map

        stem = self.stem(board)                              # (B, C, H, W)
        flat = stem.reshape(b, c, -1)                        # (B, C, H*W)
        tok_feat = flat.index_select(2, self._token_cells)   # (B, C, T)
        tok_feat = tok_feat.permute(0, 2, 1)                 # (B, T, C)

        x = (
            self.tok_in(tok_feat)
            + self.type_emb(self._token_type).unsqueeze(0)
            + self.id_emb(self._token_ids).unsqueeze(0)
        )                                                    # (B, T, d_model)
        x = self.encoder(x, mask=self._attn_mask)            # (B, T, d_model)
        delta = self.tok_out(x)                              # (B, T, C)

        # Scatter adjacency-refined token features back onto the raster grid.
        feat_map = flat.index_add(2, self._token_cells, delta.permute(0, 2, 1))
        feat_map = feat_map.reshape(b, c, *self.map_shape[1:])

        g = th.cat([delta.mean(dim=1), self.numeric_net(numeric)], dim=1)  # (B, d_g)
        return th.cat([feat_map.reshape(b, -1), g], dim=1)
