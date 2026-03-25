"""读取 CSV、清洗、标准化、划分训练/验证集；可写出 processed 文件。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 演示数据集列名（与 data/raw/sample_core_tabular.csv 一致；可替换为公开 CSV 时改此处或传参）
DEFAULT_FEATURE_COLUMNS: List[str] = ["DEPTH", "GR", "RHOB", "NPHI", "DT"]
DEFAULT_TARGET_COLUMNS: List[str] = ["PHIT", "PERM"]


def load_and_prepare(
    csv_path: str | Path,
    feature_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    perm_log10: bool = True,
    perm_eps: float = 1e-4,
    scale_targets: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StandardScaler,
    Optional[StandardScaler],
    List[str],
]:
    """
    返回标准化后的 X_train/X_val、y_train/y_val（张量）、拟合在训练集上的 X_scaler、
    可选的 y_scaler（对 y 各列单独标准化，避免多目标 MSE 尺度失衡）、特征列名。
    第二目标 PERM 默认用 log10(PERM + perm_eps) 以稳定训练。
    """
    feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
    target_columns = target_columns or list(DEFAULT_TARGET_COLUMNS)

    path = Path(csv_path)
    df = pd.read_csv(path)
    need = feature_columns + target_columns
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，已有: {list(df.columns)}")

    df = df[need].dropna()
    if len(df) < 10:
        raise ValueError("有效样本过少，请检查 CSV")

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df[target_columns].to_numpy(dtype=np.float64).copy()
    if perm_log10 and "PERM" in target_columns:
        pi = target_columns.index("PERM")
        y[:, pi] = np.log10(np.maximum(y[:, pi], 0.0) + perm_eps)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)

    y_scaler: Optional[StandardScaler] = None
    if scale_targets:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    return X_train_t, X_val_t, y_train_t, y_val_t, x_scaler, y_scaler, feature_columns


def save_processed_split(
    csv_path: str | Path,
    out_dir: str | Path,
    feature_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    perm_log10: bool = True,
    perm_eps: float = 1e-4,
) -> None:
    """写出 train.csv / val.csv（特征已标准化；PERM 先 log10 再与 PHIT 一起做 y 标准化）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
    target_columns = target_columns or list(DEFAULT_TARGET_COLUMNS)

    path = Path(csv_path)
    df = pd.read_csv(path)
    need = feature_columns + target_columns
    df = df[need].dropna()
    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df[target_columns].to_numpy(dtype=np.float64).copy()
    if perm_log10 and "PERM" in target_columns:
        pi = target_columns.index("PERM")
        y[:, pi] = np.log10(np.maximum(y[:, pi], 0.0) + perm_eps)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)

    train_df = pd.DataFrame(X_train_s, columns=feature_columns)
    val_df = pd.DataFrame(X_val_s, columns=feature_columns)
    for i, name in enumerate(target_columns):
        train_df[name] = y_train[:, i]
        val_df[name] = y_val[:, i]

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
