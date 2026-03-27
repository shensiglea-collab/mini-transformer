# 本仓库说明

本仓库包含**两个相互独立的子项目**（无共享 Python 包路径依赖）：

| 目录 | 内容 |
|------|------|
| **`experiments/`** | Boston 房价预测：多种模型对比（线性/MLP/Transformer/sklearn 等），源码在 `experiments/src/`，数据在 `experiments/data/`。详见 [experiments/README.md](experiments/README.md)。 |
| **`reservoir-property-llm/`** | 储层物性：结构化 CSV + 小 Transformer 表格回归（课题向），详见 [reservoir-property-llm/README.md](reservoir-property-llm/README.md)。 |

---

# 子项目一：`experiments/`（房价预测）

使用多种不同模型预测房价：线性回归、MLP（多层感知机）、Transformer、决策树、支持向量机、K近邻、随机森林。

## 项目结构

```
experiments/
├── src/                          # 源代码（仅此子项目使用）
│   ├── data/
│   ├── models/
│   └── training/
├── data/
│   └── housing.data.txt          # Boston Housing
├── train_all_models.py
├── train_single_model.py
├── compare_regression_models.py
└── README.md
```

## 快速开始

### 1. 训练所有模型并对比

```bash
python experiments/train_all_models.py
```

这将依次训练8个模型并输出对比结果：
- 线性回归 (基准)
- MLP (2层)
- 深层MLP (3层)
- Transformer
- 决策树
- 支持向量机 (SVM)
- K近邻 (KNN)
- 随机森林

### 2. 训练单个模型

```bash
# 训练MLP模型
python experiments/train_single_model.py --model mlp

# 训练Transformer模型
python experiments/train_single_model.py --model transformer

# 自定义训练参数
python experiments/train_single_model.py --model deep_mlp --epochs 200 --lr 0.0001
```

可用模型选项：`linear`, `mlp`, `deep_mlp`, `transformer`

## 模型说明

### 1. 线性回归 (Linear Regression)
- **用途**: 作为基准模型
- **结构**: 单层线性变换
- **参数量**: ~14

### 2. MLP (多层感知机)
- **结构**: 输入 → 64 → 64 → 输出
- **激活函数**: ReLU
- **Dropout**: 0.2
- **参数量**: ~5,000

### 3. 深层MLP (Deep MLP)
- **结构**: 输入 → 128 → 64 → 32 → 输出
- **激活函数**: ReLU
- **Dropout**: 0.3
- **参数量**: ~13,000

### 4. Transformer
- **结构**: 输入投影 → Transformer编码器 → 输出
- **d_model**: 32
- **num_heads**: 2
- **num_layers**: 1
- **参数量**: ~7,000

### 5. 决策树 (Decision Tree)
- **用途**: 基于特征分割的树形模型
- **优势**: 易解释、可处理非线性关系
- **参数**: max_depth=10

### 6. 支持向量机 (SVM)
- **用途**: 最大化间隔的分类/回归模型
- **核函数**: RBF
- **参数**: C=1.0

### 7. K近邻 (KNN)
- **用途**: 基于最近邻的回归模型
- **邻居数**: 5
- **距离**: 欧几里得距离

### 8. 随机森林 (Random Forest)
- **用途**: 多个决策树的集成模型
- **树数**: 100
- **每棵树深度**: 10

## 数据集

**Boston Housing Dataset**
- 样本数: 506
- 特征数: 13
- 目标: 房价 (单位: 千美元)

## 评估指标

- **MSE** (均方误差)
- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **R²** (决定系数)

## 代码示例

```python
from src.data import load_housing_data
from src.models import MLPModel
from src.training import Trainer

# 加载数据（若在 experiments/ 下作为 cwd，用相对路径；否则写绝对路径）
dataset = load_housing_data('experiments/data/housing.data.txt')

# 创建模型
model = MLPModel(input_size=13, hidden_size=64, output_size=1)

# 训练
trainer = Trainer(model, lr=0.001)
trainer.train(dataset.X_train_tensor, dataset.y_train_tensor, num_epochs=100)

# 评估
report = trainer.get_full_report(...)
print(f"Test R²: {report['test']['r2']:.4f}")
```

## 依赖

```
torch
numpy
scikit-learn
```

## 旧代码

原始代码保留在以下目录（供参考）：
- `models/`: 原始模型定义
- `scripts/`: 原始训练脚本
- `utils.py`: 原始工具函数

### Training Housing Models
```bash
# Simple MLP
python scripts/train_mlp.py

# Deep MLP
python scripts/train_deep_mlp.py

# Transformer
python scripts/train_transformer.py
```

### Testing
```bash
# Test Transformer functionality
python -c "from models.transformer.test_transformer import test_transformer; test_transformer()"
```

## Dependencies
- PyTorch
- scikit-learn
- numpy
- pandas

### Testing Transformer
```bash
python scripts/test.py
```

## Dependencies
- PyTorch
- scikit-learn
- numpy
- pandas