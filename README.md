# 房价预测项目

使用多种不同模型预测房价：线性回归、MLP（多层感知机）、Transformer。

## 项目结构

```
mini-transformer/
├── src/                          # 源代码目录
│   ├── data/                     # 数据模块
│   │   ├── __init__.py
│   │   ├── dataset.py            # 数据集类 (HousingDataset)
│   │   └── advanced_dataset.py   # 高级数据集类
│   ├── models/                   # 模型模块
│   │   ├── __init__.py
│   │   ├── linear_regression.py  # 线性回归模型
│   │   ├── linear_regression_v2.py  # 线性回归模型v2
│   │   ├── mlp.py                # MLP模型 (简单 + 深层)
│   │   └── transformer.py        # Transformer回归模型
│   └── training/                 # 训练模块
│       ├── __init__.py
│       ├── trainer.py            # 通用训练器 (Trainer)
│       ├── advanced_trainer.py   # 高级训练器
│       └── metrics.py            # 评估指标计算
├── experiments/                  # 实验脚本
│   ├── train_all_models.py       # 训练并对比所有模型
│   ├── train_single_model.py     # 训练单个模型
│   └── compare_regression_models.py  # 回归模型对比
├── data/                         # 数据文件
│   └── housing.data.txt          # Boston Housing数据集
└── README.md                     # 本文件
```

## 快速开始

### 1. 训练所有模型并对比

```bash
python experiments/train_all_models.py
```

这将依次训练4个模型并输出对比结果：
- 线性回归 (基准)
- MLP (2层)
- 深层MLP (3层)
- Transformer

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

# 加载数据
dataset = load_housing_data('data/housing.data.txt')

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