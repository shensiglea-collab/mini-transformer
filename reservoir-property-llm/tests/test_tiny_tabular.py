"""参数量与数据形状冒烟测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reservoir_llm.models.tiny_tabular_transformer import (  # noqa: E402
    TinyMLP,
    TinyTabularTransformer,
    count_parameters,
)


def test_transformer_under_100k_params():
    m = TinyTabularTransformer(n_features=5, n_targets=2, d_model=32, nhead=2, dim_feedforward=64)
    n = count_parameters(m)
    assert n < 100_000
    x = torch.randn(4, 5)
    y = m(x)
    assert y.shape == (4, 2)


def test_mlp_forward():
    m = TinyMLP(5, 2, hidden=64)
    y = m(torch.randn(8, 5))
    assert y.shape == (8, 2)
