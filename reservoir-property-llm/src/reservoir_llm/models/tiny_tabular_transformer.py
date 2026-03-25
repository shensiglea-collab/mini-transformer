"""表格特征：每列一个 token，经单层 TransformerEncoder 后池化回归。"""
from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class TinyTabularTransformer(nn.Module):
    """
    参数量目标：< 100k（在 n_features~5、d_model=32、1 层时远低于上限）。
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int = 2,
        d_model: int = 32,
        nhead: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.n_features = n_features
        self.d_model = d_model
        self.feature_linears = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_features)]
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(d_model, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        batch = x.size(0)
        tokens = []
        for i in range(self.n_features):
            xi = x[:, i : i + 1]
            tokens.append(self.feature_linears[i](xi))
        h = torch.stack(tokens, dim=1)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class TinyMLP(nn.Module):
    """小 MLP 基线（扁平输入）。"""

    def __init__(self, n_features: int, n_targets: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
