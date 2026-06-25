"""Cross-Dimensional (Xdim) trunk — drop-in for HexEncoder/HexFormer.

Idea from Gendre & Kaneko 2020, "Playing Catan with Cross-Dimensional Neural
Network" (arXiv 2008.07079). Catan state mixes a 2D board with scalar player
state. Our HexEncoder/HexFormer run those two pathways separately and only glue
them at the END (conv map + numeric->global vector concatenated). Xdim instead
lets the two pathways EXCHANGE information at every block:

  * INFLATION  : scalar vector -> spatial channels filled with those values, so
                 the convolutions "see" hand size / phase / robber / opponent
                 achievements while processing the board.
  * DEFLATION  : spatial channels -> a few scalars (per-channel mean + variance),
                 so the scalar pathway "sees" a summary of the board.

So a settlement spot can be scored *given my hand and the robber*, and the
global summary is *board-aware* -- exactly the board x player-state interaction a
good value/evaluator needs (our diagnosed weak link in search).

Drop-in contract (identical to HexFormer): exposes ``map_shape`` (C, H, W) and
``d_g`` and returns the flat ``[ feat_map.flatten(1) , g ]`` the spatial pointer
head / maritime head / value head consume unchanged.
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Final feature-map channels gathered per cell by the pointer head (matches HexFormer).
MAP_CHANNELS = 64


def _deflate(s: th.Tensor) -> th.Tensor:
    """Spatial (B,C,H,W) -> scalars (B,2C): per-channel mean + variance."""
    flat = s.flatten(2)                      # (B, C, H*W)
    return th.cat([flat.mean(dim=2), flat.var(dim=2)], dim=1)


class XdimBlock(nn.Module):
    """One cross-dimensional block: deflate spatial->scalar, update both, inflate
    scalar->spatial."""

    def __init__(self, c_spatial: int, h_v: int, k_inflate: int = 16):
        super().__init__()
        self.k = k_inflate
        # scalar update sees the deflated board (mean+var per spatial channel)
        self.v_net = nn.Sequential(
            nn.Linear(h_v + 2 * c_spatial, h_v), nn.LayerNorm(h_v), nn.ReLU(inplace=True),
        )
        self.v_to_chan = nn.Linear(h_v, k_inflate)        # scalars -> inflate channels
        # spatial update sees the inflated scalars as extra channels
        self.conv = nn.Sequential(
            nn.Conv2d(c_spatial + k_inflate, c_spatial, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, s: th.Tensor, v: th.Tensor):
        b, _, h, w = s.shape
        v = self.v_net(th.cat([v, _deflate(s)], dim=1))                  # scalar <- board
        infl = self.v_to_chan(v)[:, :, None, None].expand(b, self.k, h, w)  # scalar -> board
        s = self.conv(th.cat([s, infl], dim=1))
        return s, v


class XdimTrunk(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        d_g: int = 256,
        c_spatial: int = 48,
        h_v: int = 192,
        n_blocks: int = 3,
        k_inflate: int = 16,
    ):
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]
        in_ch, height, width = (int(s) for s in board_space.shape)
        n_numeric = int(numeric_space.shape[0])
        c_map = MAP_CHANNELS

        n_map = c_map * height * width
        super().__init__(observation_space, features_dim=n_map + d_g)
        self.map_shape = (c_map, height, width)
        self.d_g = d_g
        self._c_map = c_map

        # stems
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_ch, c_spatial, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.numeric_stem = nn.Sequential(
            nn.LayerNorm(n_numeric), nn.Linear(n_numeric, h_v), nn.ReLU(inplace=True),
        )
        # cross-dimensional blocks
        self.blocks = nn.ModuleList(
            XdimBlock(c_spatial, h_v, k_inflate) for _ in range(n_blocks)
        )
        # heads: final feature map + board-aware global vector
        self.to_map = nn.Conv2d(c_spatial, c_map, kernel_size=3, padding=1)
        self.to_g = nn.Sequential(
            nn.Linear(h_v + 2 * c_spatial, d_g), nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        board = observations["board"].float()
        numeric = observations["numeric"].float()
        b = board.shape[0]

        s = self.conv_stem(board)               # (B, c_spatial, H, W)
        v = self.numeric_stem(numeric)          # (B, h_v)
        for block in self.blocks:
            s, v = block(s, v)

        feat_map = self.to_map(s)               # (B, c_map, H, W)
        g = self.to_g(th.cat([v, _deflate(s)], dim=1))   # (B, d_g), board-aware
        return th.cat([feat_map.reshape(b, -1), g], dim=1)
