# Boston Housing 多模型实验（mini-transformer）

与仓库根目录下的 `reservoir-property-llm/` **相互独立**：本目录自包含 `src/` 与 `data/`。

在**仓库根目录**执行：

```bash
python experiments/train_all_models.py
python experiments/train_single_model.py --model mlp
python experiments/compare_regression_models.py
```

数据路径：`experiments/data/housing.data.txt`。
