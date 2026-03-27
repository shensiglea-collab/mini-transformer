"""
统一训练脚本 - 训练并对比所有模型
运行: 
  python experiments/train_all_models.py
  python experiments/train_all_models.py --transformer large
  python experiments/train_all_models.py --transformer all
"""
import sys
import os
import argparse

# 源码与数据均在 experiments/ 下，与仓库根目录的 reservoir-property-llm 相互独立
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS_DIR)

from src.data import load_housing_data
from src.models import (
    LinearRegressionModel, MLPModel, DeepMLPModel, TransformerRegressor,
    DecisionTreeModel, SVMModel, KNNModel, RandomForestModel,
    LinearRegressionL2, LinearRegressionL1, LinearRegressionElasticNet, ImprovedLinearRegression
)
from src.training import Trainer
from src.training.metrics import MetricsCalculator, print_predictions


# Transformer 预设配置
TRANSFORMER_PRESETS = {
    'small': {
        'model_args': {'preset': 'small'},
        'train_config': {'epochs': 100, 'batch_size': 32, 'lr': 0.001},
        'display_name': 'Transformer-Small',
        'description': 'd_model=32, heads=2, layers=1'
    },
    'medium': {
        'model_args': {'preset': 'medium'},
        'train_config': {'epochs': 150, 'batch_size': 32, 'lr': 0.0005},
        'display_name': 'Transformer-Medium',
        'description': 'd_model=64, heads=4, layers=2'
    },
    'large': {
        'model_args': {'preset': 'large'},
        'train_config': {'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
        'display_name': 'Transformer-Large',
        'description': 'd_model=128, heads=4, layers=3'
    },
    'xlarge': {
        'model_args': {'preset': 'xlarge'},
        'train_config': {'epochs': 300, 'batch_size': 16, 'lr': 0.00005},
        'display_name': 'Transformer-XLarge',
        'description': 'd_model=256, heads=8, layers=4'
    }
}


def run_experiment(model, model_name, dataset, config):
    """
    运行单个模型的实验

    Args:
        model: 模型实例
        model_name: 模型名称
        dataset: 数据集实例
        config: 训练配置

    Returns:
        测试集指标
    """
    print(f"\n{'='*70}")
    print(f"🚀 {model_name}")
    print(f"{'='*70}")

    # 模型信息
    model_info = model.get_model_info()
    if model_info.get('model_type') == 'sklearn':
        print(f"\n📊 模型信息:")
        print(f"   类型: sklearn")
        print(f"   模型: {model_info['model_name']}")
    else:
        print(f"\n📊 模型信息:")
        if hasattr(model, 'config'):
            print(f"   d_model: {model.config.d_model}")
            print(f"   num_heads: {model.config.num_heads}")
            print(f"   num_layers: {model.config.num_layers}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 获取numpy数据
    y_train, y_test = dataset.get_numpy_data()
    X_train_np = dataset.X_train
    X_test_np = dataset.X_test

    if model_info.get('model_type') == 'sklearn':
        # sklearn模型训练
        print(f"\n🔥 开始训练 sklearn 模型...")
        model.fit(X_train_np, y_train)

        # 预测
        train_pred = model.predict(X_train_np)
        test_pred = model.predict(X_test_np)

    else:
        # PyTorch模型训练
        trainer = Trainer(
            model=model,
            model_name=model_name,
            lr=config['lr'],
            device=config.get('device', 'cpu')
        )

        # 训练
        print(f"\n🔥 开始训练 (epochs={config['epochs']}, batch_size={config['batch_size']})...")
        trainer.train(
            dataset.X_train_tensor,
            dataset.y_train_tensor,
            num_epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=True
        )

        # 获取完整报告
        report = trainer.get_full_report(
            dataset.X_train_tensor, dataset.y_train_tensor,
            dataset.X_test_tensor, dataset.y_test_tensor,
            y_train, y_test
        )

        train_pred = report['train_pred']
        test_pred = report['test_pred']

    # 评估
    print(f"\n📈 模型评估:")
    train_metrics = MetricsCalculator.calculate(y_train, train_pred)
    test_metrics = MetricsCalculator.calculate(y_test, test_pred)

    MetricsCalculator.print_metrics(train_metrics, "训练集")
    MetricsCalculator.print_metrics(test_metrics, "测试集")

    # 计算参数数量
    if model_info.get('model_type') == 'sklearn':
        # sklearn模型的参数计数（近似值）
        try:
            # 尝试获取模型参数
            if hasattr(model, 'coef_'):
                param_count = len(model.coef_) + (1 if hasattr(model, 'intercept_') else 0)
            else:
                param_count = 0  # 无法确定参数数量
        except:
            param_count = 0
    else:
        # PyTorch模型的参数计数
        param_count = sum(p.numel() for p in model.parameters())

    # 预测示例
    print_predictions(y_test, test_pred, num_samples=5)

    print(f"\n{'='*70}")

    return test_metrics, param_count


def main():
    """主函数 - 运行所有模型的对比实验"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='训练并对比所有房价预测模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python experiments/train_all_models.py                     # 使用 small transformer
  python experiments/train_all_models.py --transformer large # 使用 large transformer
  python experiments/train_all_models.py --transformer all   # 对比所有 transformer 配置

Transformer 预设配置:
  small:  d_model=32,  heads=2,  layers=1,  ~9K params
  medium: d_model=64,  heads=4,  layers=2,  ~68K params
  large:  d_model=128, heads=4,  layers=3,  ~400K params
  xlarge: d_model=256, heads=8,  layers=4,  ~2.1M params
        """
    )
    parser.add_argument('--transformer', type=str, default='small',
                        choices=['small', 'medium', 'large', 'xlarge', 'all'],
                        help='Transformer 配置预设 (默认: small, all=对比所有配置)')
    parser.add_argument('--skip-linear', action='store_true',
                        help='跳过线性回归系列模型，加快实验速度')
    
    args = parser.parse_args()
    
    args.transformer = 'all'

    print("\n" + "="*70)
    print("🏠 房价预测模型对比实验")
    print("   数据集: Boston Housing")
    if args.transformer == 'all':
        print("   Transformer: 对比所有配置 (small/medium/large/xlarge)")
    else:
        preset_info = TRANSFORMER_PRESETS[args.transformer]
        print(f"   Transformer: {args.transformer} ({preset_info['description']})")
    print("="*70)

    # ==================== 加载数据 ====================
    print("\n📦 加载数据...")
    data_path = os.path.join(_EXPERIMENTS_DIR, 'data', 'housing.data.txt')
    dataset = load_housing_data(data_path)
    dataset.print_info()

    # ==================== 模型配置 ====================
    configs = {
        'linear': {'epochs': 200, 'batch_size': 32, 'lr': 0.01},
        'mlp': {'epochs': 100, 'batch_size': 32, 'lr': 0.001},
        'deep_mlp': {'epochs': 150, 'batch_size': 32, 'lr': 0.0005},
        'sklearn': {'epochs': None, 'batch_size': None, 'lr': None}
    }

    # ==================== 运行实验 ====================
    results = {}
    param_counts = {}

    # 1. 线性回归系列
    if not args.skip_linear:
        model_linear = LinearRegressionModel(input_size=13, output_size=1)
        test_metrics, param_count = run_experiment(model_linear, "线性回归 (基准)", dataset, configs['linear'])
        results['线性回归'] = test_metrics
        param_counts['线性回归'] = param_count

        model_linear_l2 = LinearRegressionL2(input_size=13, output_size=1, weight_decay=0.01)
        test_metrics, param_count = run_experiment(model_linear_l2, "线性回归 (L2正则化)", dataset, configs['linear'])
        results['线性回归(L2)'] = test_metrics
        param_counts['线性回归(L2)'] = param_count

        model_linear_l1 = LinearRegressionL1(input_size=13, output_size=1, l1_lambda=0.01)
        test_metrics, param_count = run_experiment(model_linear_l1, "线性回归 (L1正则化)", dataset, configs['linear'])
        results['线性回归(L1)'] = test_metrics
        param_counts['线性回归(L1)'] = param_count

        model_linear_elastic = LinearRegressionElasticNet(input_size=13, output_size=1, l1_lambda=0.01, l2_lambda=0.01)
        test_metrics, param_count = run_experiment(model_linear_elastic, "线性回归 (Elastic Net)", dataset, configs['linear'])
        results['线性回归(ElasticNet)'] = test_metrics
        param_counts['线性回归(ElasticNet)'] = param_count

        model_linear_improved = ImprovedLinearRegression(input_size=13, output_size=1, dropout=0.1, use_bn=True)
        test_metrics, param_count = run_experiment(model_linear_improved, "改进版线性回归", dataset, configs['linear'])
        results['改进线性回归'] = test_metrics
        param_counts['改进线性回归'] = param_count

    # 2. MLP
    model_mlp = MLPModel(input_size=13, hidden_size=64, output_size=1)
    test_metrics, param_count = run_experiment(model_mlp, "MLP (2层)", dataset, configs['mlp'])
    results['MLP'] = test_metrics
    param_counts['MLP'] = param_count

    model_deep_mlp = DeepMLPModel(input_size=13, hidden_sizes=[128, 64, 32], output_size=1)
    test_metrics, param_count = run_experiment(model_deep_mlp, "深层MLP (3层)", dataset, configs['deep_mlp'])
    results['深层MLP'] = test_metrics
    param_counts['深层MLP'] = param_count


    # 3. Transformer
    if args.transformer == 'all':
        # 对比所有 Transformer 配置
        for preset_name, preset_config in TRANSFORMER_PRESETS.items():
            model = TransformerRegressor.from_preset(preset_name, input_size=13)
            display_name = preset_config['display_name']
            test_metrics, param_count = run_experiment(
                model, 
                f"{display_name} ({preset_config['description']})", 
                dataset, 
                preset_config['train_config']
            )
            results[display_name] = test_metrics
            param_counts[display_name] = param_count
    else:
        # 使用指定的 Transformer 配置
        preset_config = TRANSFORMER_PRESETS[args.transformer]
        model = TransformerRegressor.from_preset(args.transformer, input_size=13)
        test_metrics, param_count = run_experiment(
            model, 
            f"Transformer ({preset_config['description']})", 
            dataset, 
            preset_config['train_config']
        )
        results['Transformer'] = test_metrics
        param_counts['Transformer'] = param_count

    # 4. sklearn 模型
    model_dt = DecisionTreeModel(max_depth=10)
    test_metrics, param_count = run_experiment(model_dt, "决策树", dataset, configs['sklearn'])
    results['决策树'] = test_metrics
    param_counts['决策树'] = param_count

    model_svm = SVMModel(kernel='rbf', C=1.0)
    test_metrics, param_count = run_experiment(model_svm, "支持向量机 (SVM)", dataset, configs['sklearn'])
    results['SVM'] = test_metrics
    param_counts['SVM'] = param_count

    model_knn = KNNModel(n_neighbors=5)
    test_metrics, param_count = run_experiment(model_knn, "K近邻 (KNN)", dataset, configs['sklearn'])
    results['KNN'] = test_metrics
    param_counts['KNN'] = param_count

    model_rf = RandomForestModel(n_estimators=100, max_depth=10)
    test_metrics, param_count = run_experiment(model_rf, "随机森林", dataset, configs['sklearn'])
    results['随机森林'] = test_metrics
    param_counts['随机森林'] = param_count

    # ==================== 对比结果 ====================
    print("\n" + "="*80)
    print("📊 最终对比结果")
    print("="*80)
    MetricsCalculator.print_comparison(results, param_counts)

    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 最佳模型 (按MSE): {best_model[0]}")
    print(f"   MSE: {best_model[1]['mse']:.4f}, R²: {best_model[1]['r2']:.4f}")

    print("\n" + "="*80)
    print("✅ 所有实验完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
