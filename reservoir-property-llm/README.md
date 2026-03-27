# 储层物性 — 结构化数据 + 小 Transformer 实验

在**表格数据**上训练参数量 **&lt; 10 万** 的小型 Transformer 回归模型（孔隙度 + log10 渗透率），与课题里的大模型 SFT 样本分离。

与上级目录中的 **`experiments/`（Boston 房价）项目相互独立**，无代码依赖。

## 结构

```
reservoir-property-llm/
├── data/raw/              # 原始 CSV（含演示 sample_core_tabular.csv）
├── data/processed/        # 可选：train.csv / val.csv
├── scripts/
│   ├── build_demo_csv.py
│   └── train_tiny_transformer.py
├── src/reservoir_llm/
│   ├── data/load_tabular.py
│   ├── data/README-data.md   # 列说明与命令
│   └── models/tiny_tabular_transformer.py
├── tests/
└── requirements.txt
```

## 快速运行

```bash
cd reservoir-property-llm
pip install -r requirements.txt
python scripts/build_demo_csv.py    # 若尚无 sample_core_tabular.csv
python scripts/train_tiny_transformer.py --save-processed data/processed
```

课题用 8 条 LLM 样本见 `data/raw/selected_val_samples.txt`。
