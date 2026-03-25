# 结构化孔渗数据说明

## 当前演示数据（默认）

- **文件**：[`data/raw/sample_core_tabular.csv`](../../data/raw/sample_core_tabular.csv)  
- **性质**：**程序合成的演示表**，用于验证「读取 → 划分 → 标准化 → 小 Transformer 训练」全流程，**不代表真实岩心**。  
- **生成命令**（仓库根为 `reservoir-property-llm/`）：

```bash
python scripts/build_demo_csv.py -n 800 -o data/raw/sample_core_tabular.csv
```

## 列定义（默认 schema）

| 列名 | 角色 | 说明 |
|------|------|------|
| DEPTH | 特征 | 深度 (m) |
| GR | 特征 | 自然伽马 (API 量级) |
| RHOB | 特征 | 体积密度 |
| NPHI | 特征 | 中子孔隙度 (v/v) |
| DT | 特征 | 声波时差 (us/ft 量级) |
| PHIT | **目标** | 总孔隙度 (%) |
| PERM | **目标** | 空气渗透率 (mD)；训练时内部转为 `log10(PERM + 1e-4)` |

代码中默认列名常量见 [`load_tabular.py`](load_tabular.py) 的 `DEFAULT_FEATURE_COLUMNS` / `DEFAULT_TARGET_COLUMNS`。

## 换成公开/业务 CSV 时

1. 将文件放入 `data/raw/`，保证包含上述列名（或修改 `load_tabular.py` 中的默认列表，或后续扩展 CLI 传参）。  
2. 对渗透率使用对数目标可减轻长尾，本管线默认已开启。  
3. 运行训练前可选写出处理后的划分：

```bash
python scripts/train_tiny_transformer.py --csv data/raw/your.csv --save-processed data/processed
```

将生成 `data/processed/train.csv` 与 `val.csv`（特征已按**训练集**标准化；目标列为 **log10(PERM) 与 PHIT 经训练集拟合的 StandardScaler 后的值**，推理评估时需用同一 `y_scaler` 反变换——训练脚本内部已处理）。

## 训练命令

```bash
pip install -r requirements.txt
python scripts/train_tiny_transformer.py
python scripts/train_tiny_transformer.py --epochs 200 --lr 0.001 --save-processed data/processed
```

输出包含：**sklearn 线性回归**、**小 MLP**、**TinyTabularTransformer** 的验证集 MAE / R²（在 **PHIT 与 log10(PERM) 的原始尺度**上计算，训练中对 `y` 标准化以平衡多目标 MSE），并打印 Transformer **参数量（须 < 100000）**。

## 与课题 SFT 样本的关系

[`data/raw/selected_val_samples.txt`](../../data/raw/selected_val_samples.txt) 中的 8 条大模型样本**不参与**本表格实验训练；本阶段目标是打通**结构化监督学习**。后续可将薄片/XRD 解析为表格列再对齐同一管线。
