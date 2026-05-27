"""CNN feature extractor for Catanatron's `representation="mixed"` Dict obs.

Board branch: 3 conv layers (3x3, same-padding, ReLU) -> global average pool ->
linear -> ReLU. Numeric branch: 2-layer MLP. Concat -> 512-dim features.
~600-800k params depending on input channels (16 for 1v1, 20 for 4p).
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BoardCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        features_dim: int = 512,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        numeric_hidden: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]

        in_ch = board_space.shape[0]
        c1, c2, c3 = conv_channels
        self.board_net = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, features_dim // 2),
            nn.ReLU(inplace=True),
        )

        n_numeric = int(numeric_space.shape[0])
        self.numeric_net = nn.Sequential(
            nn.Linear(n_numeric, numeric_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(numeric_hidden, features_dim // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        board = observations["board"].float()
        numeric = observations["numeric"].float()
        return th.cat([self.board_net(board), self.numeric_net(numeric)], dim=1)
