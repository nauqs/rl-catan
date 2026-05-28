"""Feature extractors for Catanatron's `representation="mixed"` Dict obs.

Three variants — register them in `catan_rl/features/__init__.py:EXTRACTORS`:

  - `BoardCNN`      : 3 conv layers -> GAP -> linear. ~700k params. Phase 1 default.
  - `BoardCNNAttn`  : Same conv stack + one transformer encoder layer over the
                      spatial positions before pooling. ~900k–1.1M params.
  - `MLPExtractor`  : Flatten board + numeric, dense MLP. ~5M params. Ablation.
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


class BoardCNNAttn(BaseFeaturesExtractor):
    """BoardCNN + one transformer encoder layer over the spatial positions."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        features_dim: int = 512,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        nhead: int = 4,
        dim_feedforward: int = 256,
        numeric_hidden: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]

        in_ch = board_space.shape[0]
        c1, c2, c3 = conv_channels
        self.board_conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c3, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0,
        )
        self.board_attn = encoder_layer
        self.board_proj = nn.Linear(c3, features_dim // 2)

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
        # Conv -> (B, C, H, W). Reshape to (B, H*W, C) for attention.
        feats = self.board_conv(board)
        b, c, h, w = feats.shape
        seq = feats.view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        seq = self.board_attn(seq)                      # (B, H*W, C)
        pooled = seq.mean(dim=1)                        # (B, C)
        board_out = th.relu(self.board_proj(pooled))    # (B, features_dim//2)
        return th.cat([board_out, self.numeric_net(numeric)], dim=1)


class MLPExtractor(BaseFeaturesExtractor):
    """Flatten board + numeric, dense MLP. Pure ablation against the conv prior."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        features_dim: int = 512,
        hidden_layers: tuple[int, ...] = (1024, 512, 256),
    ):
        super().__init__(observation_space, features_dim=features_dim)
        board_space: gym.spaces.Box = observation_space.spaces["board"]
        numeric_space: gym.spaces.Box = observation_space.spaces["numeric"]

        n_board = 1
        for s in board_space.shape:
            n_board *= s
        n_numeric = int(numeric_space.shape[0])
        in_dim = n_board + n_numeric

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, features_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        board = observations["board"].float().flatten(start_dim=1)
        numeric = observations["numeric"].float()
        return self.net(th.cat([board, numeric], dim=1))
