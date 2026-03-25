#!/usr/bin/env python3
"""
训练： sklearn 线性回归基线 + 小 MLP + TinyTabularTransformer。
默认使用 data/raw/sample_core_tabular.csv（可先运行 scripts/build_demo_csv.py）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from reservoir_llm.data.load_tabular import (  # noqa: E402
    load_and_prepare,
    save_processed_split,
)
from reservoir_llm.models.tiny_tabular_transformer import (  # noqa: E402
    TinyMLP,
    TinyTabularTransformer,
    count_parameters,
)


def train_torch(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vpred = model(X_val)
                vloss = loss_fn(vpred, y_val).item()
            print(f"  epoch {ep+1}/{epochs}  train_loss={loss.item():.6f}  val_loss={vloss:.6f}")


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    print(f"  [{name}] MAE (per target) = {mae}  R2 = {r2}")


def to_original_y(arr: np.ndarray, y_scaler) -> np.ndarray:
    if y_scaler is None:
        return arr
    return y_scaler.inverse_transform(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "data" / "raw" / "sample_core_tabular.csv",
        help="原始 CSV 路径",
    )
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save-processed",
        type=Path,
        default=None,
        help="若指定，将标准化后的 train/val 写入该目录",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"未找到 {args.csv}，正在生成演示 CSV …")
        import subprocess

        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "build_demo_csv.py")],
            check=True,
            cwd=str(_ROOT),
        )

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    X_train, X_val, y_train, y_val, _x_scaler, y_scaler, feat_names = load_and_prepare(
        args.csv
    )
    n_feat = X_train.shape[1]
    n_tgt = y_train.shape[1]
    print(f"特征数={n_feat} {feat_names}  目标数={n_tgt}  train={len(X_train)} val={len(X_val)}")

    if args.save_processed:
        save_processed_split(args.csv, args.save_processed)

    Xtr = X_train.numpy()
    Xva = X_val.numpy()
    ytr = y_train.numpy()
    yva = y_val.numpy()
    y_va_orig = to_original_y(yva, y_scaler)

    print("\n--- sklearn LinearRegression (baseline) ---")
    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    report("val", y_va_orig, to_original_y(lin.predict(Xva), y_scaler))

    print("\n--- TinyMLP ---")
    mlp = TinyMLP(n_feat, n_tgt, hidden=64)
    print(f"  parameters = {count_parameters(mlp):,}")
    train_torch(mlp, X_train, y_train, X_val, y_val, args.epochs, args.lr, device)
    mlp.eval()
    with torch.no_grad():
        pred = mlp(X_val.to(device)).cpu().numpy()
        report("val", y_va_orig, to_original_y(pred, y_scaler))

    print("\n--- TinyTabularTransformer ---")
    tt = TinyTabularTransformer(
        n_features=n_feat,
        n_targets=n_tgt,
        d_model=32,
        nhead=2,
        dim_feedforward=64,
        dropout=0.05,
    )
    n_params = count_parameters(tt)
    print(f"  parameters = {n_params:,}")
    if n_params >= 100_000:
        raise SystemExit("参数量应 < 100000，请缩小 d_model / dim_feedforward")
    train_torch(tt, X_train, y_train, X_val, y_val, args.epochs, args.lr, device)
    tt.eval()
    with torch.no_grad():
        pred = tt(X_val.to(device)).cpu().numpy()
        report("val", y_va_orig, to_original_y(pred, y_scaler))

    print("\n说明：指标在原始目标空间计算：PHIT(%) 与 log10(PERM)；训练时对 y 做了标准化。")


if __name__ == "__main__":
    main()
