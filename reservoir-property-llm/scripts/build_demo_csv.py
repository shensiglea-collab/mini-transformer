#!/usr/bin/env python3
"""生成演示用结构化孔渗表（非真实岩心，仅用于打通训练管线）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_core_tabular.csv",
    )
    p.add_argument("-n", type=int, default=800, help="行数")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n
    depth = rng.uniform(900.0, 3200.0, n)
    gr = rng.uniform(40.0, 120.0, n)
    rhob = rng.uniform(2.2, 2.65, n)
    nphi = rng.uniform(0.1, 0.35, n)
    dt = rng.uniform(55.0, 95.0, n)

    # 可学习的伪物理关系：孔隙度(%) 与密度/中子相关
    phi_frac = (
        0.35
        - 0.12 * (rhob - 2.2) / 0.45
        + 0.2 * (nphi - 0.1) / 0.25
        - 0.00015 * (depth - 900.0) / 2300.0
        + rng.normal(0.0, 0.025, n)
    )
    phi_frac = np.clip(phi_frac, 0.04, 0.38)
    phit = phi_frac * 100.0

    logk = -1.5 + 0.07 * phit + 0.008 * (gr - 75.0) + rng.normal(0.0, 0.35, n)
    perm = np.power(10.0, logk)
    perm = np.clip(perm, 0.01, 8000.0)

    df = pd.DataFrame(
        {
            "DEPTH": depth,
            "GR": gr,
            "RHOB": rhob,
            "NPHI": nphi,
            "DT": dt,
            "PHIT": phit,
            "PERM": perm,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"写入 {n} 行 -> {args.output}")


if __name__ == "__main__":
    main()
