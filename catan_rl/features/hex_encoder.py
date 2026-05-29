"""Position-preserving feature extractor for the spatial pointer policy.

Unlike `BoardCNN` / `BoardCNNAttn` (which pool the board into one vector and
throw away *where* things are), `HexEncoder` keeps the full conv feature map so
the action head can read the local features *at* each candidate node/edge/tile.

SB3's pipeline expects a feature extractor to return a single flat tensor, which
then feeds the policy/value MLPs. To keep save/load + `predict()` preprocessing
working unchanged, we return ``concat[ flatten(map) , g ]`` and let
``SpatialMaskableActorCriticPolicy`` reshape it back into ``(map, g)``:

    map = feats[:, : C_MAP * H * W].view(B, C_MAP, H, W)
    g   = feats[:, C_MAP * H * W :]
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Output channels of the conv trunk (the spatial feature dim gathered per cell).
MAP_CHANNELS = 64


class HexEncoder(BaseFeaturesExtractor):
    """Conv trunk (no pooling) + global summary, returned as one flat tensor.

    The flat layout is ``[ map.flatten(1) , g ]`` with sizes
    ``MAP_CHANNELS * H * W`` and ``d_g`` respectively. ``map_shape`` and ``d_g``
    are exposed so the policy can split + reshape.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        d_g: int = 256,
        conv_channels: tuple[int, int, int] = (64, 64, MAP_CHANNELS),
        numeric_hidden: int = 256,
    ):
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]

        in_ch, height, width = (int(s) for s in board_space.shape)
        c1, c2, c3 = conv_channels
        assert c3 == MAP_CHANNELS, "last conv channel must equal MAP_CHANNELS"

        n_map = c3 * height * width
        super().__init__(observation_space, features_dim=n_map + d_g)

        self.map_shape = (c3, height, width)
        self.d_g = d_g
        self._n_map = n_map

        self.board_conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        n_numeric = int(numeric_space.shape[0])
        self.numeric_net = nn.Sequential(
            nn.LayerNorm(n_numeric),
            nn.Linear(n_numeric, numeric_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(numeric_hidden, d_g - c3),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        board = observations["board"].float()
        numeric = observations["numeric"].float()
        feat_map = self.board_conv(board)                       # (B, C, H, W)
        gap = feat_map.mean(dim=(2, 3))                         # (B, C)
        g = th.cat([gap, self.numeric_net(numeric)], dim=1)     # (B, d_g)
        return th.cat([feat_map.flatten(1), g], dim=1)          # (B, n_map + d_g)
