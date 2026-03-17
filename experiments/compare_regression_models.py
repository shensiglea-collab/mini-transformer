"""
回归模型对比实验
对比原始线性回归 vs 改进版（L1/L2正则化、更好的预处理）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_housing_data
from src.data.advanced_dataset import load_advanced_housing_data
from src.models.linear_regression import LinearRegressionModel
from src.models.linear_regression_v2 import (
    LinearRegressionL2, LinearRegressionL1,
    LinearRegressionElasticNet, ImprovedLinearRegression
)
from src.training import Trainer
from src.training.advanced_trainer import AdvancedTrainer
from src.training.metrics import MetricsCalculator, print_predictions


def run_experiment(model, model_name, dataset, config, use_advanced_trainer=False):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"🚀 {model_name}")
    print(f"{'='*70}")

    print(f"\n📊 模型信息:")
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        for k, v in info.items():
            print(f"   {k}: {v}")
    else:
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 选择训练器
    if use_advanced_trainer:
        trainer = AdvancedTrainer(
            model=model,
            model_name=model_name,
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.0),
            device=config.get('device', 'cpu')
        )
    else:
        trainer = Trainer(
            model=model,
            model_name=model_name,
            lr=config['lr'],
            device=config.get('device', 'cpu')
        )

    # 训练
    print(f"\n🔥 开始训练 (epochs={config['epochs']})...")
    train_kwargs = {
        'X_train': dataset.X_train_tensor,
        'y_train': dataset.y_train_tensor,
        'num_epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'verbose': True
    }
    # 只有AdvancedTrainer支持l1_lambda参数
    if use_advanced_trainer:
        train_kwargs['l1_lambda'] = config.get('l1_lambda', 0.0)
    trainer.train(**train_kwargs)

    # 评估
    print(f"\n📈 模型评估:")
    y_train, y_test = dataset.get_numpy_data()
    report = trainer.get_full_report(
        dataset.X_train_tensor, dataset.y_train_tensor,
        dataset.X_test_tensor, dataset.y_test_tensor,
        y_train, y_test
    )

    MetricsCalculator.print_metrics(report['train'], "训练集")
    MetricsCalculator.print_metrics(report['test'], "测试集")
    print_predictions(y_test, report['test_pred'], num_samples=5)

    return report['test']


def main():
    """主函数 - 对比所有回归模型"""
    print("\n" + "="*70)
    print("🏠 线性回归模型对比实验")
    print("   对比：原始模型 vs L1/L2正则化 vs 改进预处理")
    print("="*70)

    # ==================== 实验1: 原始数据 + 原始模型 ====================
    print("\n" + "="*70)
    print("实验组1: 原始数据 + 原始线性回归 (基准)")
    print("="*70)

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'housing.data.txt')
    dataset_original = load_housing_data(data_path)
    dataset_original.print_info()

    results = {}

    # 原始线性回归
    model_original = LinearRegressionModel(input_size=13, output_size=1)
    results['原始线性回归'] = run_experiment(
        model_original, "原始线性回归 (基准)",
        dataset_original, {'epochs': 200, 'batch_size': 32, 'lr': 0.01}
    )

    # ==================== 实验2: 改进数据 + 原始模型 ====================
    print("\n" + "="*70)
    print("实验组2: 改进预处理 + 原始线性回归")
    print("   - 排除B字段")
    print("   - 对CRIM/TAX进行对数变换")
    print("   - 使用RobustScaler")
    print("="*70)

    dataset_improved = load_advanced_housing_data(
        data_path,
        exclude_b=True,
        scaler_type='robust',
        add_polynomial=False
    )
    dataset_improved.print_info()

    # 改进数据 + 原始模型
    model_with_improved_data = LinearRegressionModel(
        input_size=dataset_improved.X.shape[1],
        output_size=1
    )
    results['改进数据+原始模型'] = run_experiment(
        model_with_improved_data, "改进数据 + 原始模型",
        dataset_improved, {'epochs': 200, 'batch_size': 32, 'lr': 0.01}
    )

    # ==================== 实验3: 改进数据 + L2正则化 ====================
    print("\n" + "="*70)
    print("实验组3: 改进预处理 + L2正则化 (Ridge)")
    print("="*70)

    model_l2 = LinearRegressionL2(
        input_size=dataset_improved.X.shape[1],
        output_size=1,
        weight_decay=0.01
    )
    results['L2正则化'] = run_experiment(
        model_l2, "L2正则化 (Ridge)",
        dataset_improved,
        {'epochs': 200, 'batch_size': 32, 'lr': 0.01, 'weight_decay': 0.01},
        use_advanced_trainer=True
    )

    # ==================== 实验4: 改进数据 + L1正则化 ====================
    print("\n" + "="*70)
    print("实验组4: 改进预处理 + L1正则化 (Lasso)")
    print("="*70)

    model_l1 = LinearRegressionL1(
        input_size=dataset_improved.X.shape[1],
        output_size=1,
        l1_lambda=0.001
    )
    results['L1正则化'] = run_experiment(
        model_l1, "L1正则化 (Lasso)",
        dataset_improved,
        {'epochs': 200, 'batch_size': 32, 'lr': 0.01, 'l1_lambda': 0.001},
        use_advanced_trainer=True
    )

    # ==================== 实验5: 改进数据 + ElasticNet ====================
    print("\n" + "="*70)
    print("实验组5: 改进预处理 + ElasticNet (L1+L2)")
    print("="*70)

    model_elastic = LinearRegressionElasticNet(
        input_size=dataset_improved.X.shape[1],
        output_size=1,
        l1_lambda=0.001,
        l2_lambda=0.01
    )
    results['ElasticNet'] = run_experiment(
        model_elastic, "ElasticNet",
        dataset_improved,
        {'epochs': 200, 'batch_size': 32, 'lr': 0.01, 'l1_lambda': 0.001},
        use_advanced_trainer=True
    )

    # ==================== 对比结果 ====================
    print("\n" + "="*70)
    print("📊 最终对比结果")
    print("="*70)
    MetricsCalculator.print_comparison(results)

    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 最佳模型 (按MSE): {best_model[0]}")
    print(f"   MSE: {best_model[1]['mse']:.4f}, R²: {best_model[1]['r2']:.4f}")

    print("\n" + "="*70)
    print("✅ 所有实验完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
