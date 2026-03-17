"""
统一训练脚本 - 训练并对比所有模型
运行: python experiments/train_all_models.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_housing_data
from src.models import (
    LinearRegressionModel, MLPModel, DeepMLPModel, TransformerRegressor,
    DecisionTreeModel, SVMModel, KNNModel, RandomForestModel
)
from src.training import Trainer
from src.training.metrics import MetricsCalculator, print_predictions


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

    # 预测示例
    print_predictions(y_test, test_pred, num_samples=5)

    print(f"\n{'='*70}")

    return test_metrics


def main():
    """主函数 - 运行所有模型的对比实验"""
    print("\n" + "="*70)
    print("🏠 房价预测模型对比实验")
    print("   数据集: Boston Housing")
    print("   模型: 线性回归 / MLP / 深层MLP / Transformer / 决策树 / SVM / KNN / 随机森林")
    print("="*70)

    # ==================== 加载数据 ====================
    print("\n📦 加载数据...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'housing.data.txt')
    dataset = load_housing_data(data_path)
    dataset.print_info()

    # ==================== 模型配置 ====================
    # 为不同模型定义训练配置
    configs = {
        'linear': {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.01
        },
        'mlp': {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.001
        },
        'deep_mlp': {
            'epochs': 150,
            'batch_size': 32,
            'lr': 0.0005
        },
        'transformer': {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.001
        },
        'sklearn': {  # sklearn模型不需要这些参数，但保留结构
            'epochs': None,
            'batch_size': None,
            'lr': None
        }
    }

    # ==================== 运行实验 ====================
    results = {}

    # 1. 线性回归 (基准模型)
    model_linear = LinearRegressionModel(input_size=13, output_size=1)
    results['线性回归'] = run_experiment(model_linear, "线性回归 (基准)", dataset, configs['linear'])

    # 2. 简单MLP
    model_mlp = MLPModel(input_size=13, hidden_size=64, output_size=1)
    results['MLP'] = run_experiment(model_mlp, "MLP (2层)", dataset, configs['mlp'])

    # 3. 深层MLP
    model_deep_mlp = DeepMLPModel(input_size=13, hidden_sizes=[128, 64, 32], output_size=1)
    results['深层MLP'] = run_experiment(model_deep_mlp, "深层MLP (3层)", dataset, configs['deep_mlp'])

    # 4. Transformer
    model_transformer = TransformerRegressor(
        input_size=13, d_model=32, num_heads=2, d_ff=64, num_layers=1
    )
    results['Transformer'] = run_experiment(model_transformer, "Transformer", dataset, configs['transformer'])

    # 5. 决策树
    model_dt = DecisionTreeModel(max_depth=10)
    results['决策树'] = run_experiment(model_dt, "决策树", dataset, configs['sklearn'])

    # 6. 支持向量机
    model_svm = SVMModel(kernel='rbf', C=1.0)
    results['SVM'] = run_experiment(model_svm, "支持向量机 (SVM)", dataset, configs['sklearn'])

    # 7. K近邻
    model_knn = KNNModel(n_neighbors=5)
    results['KNN'] = run_experiment(model_knn, "K近邻 (KNN)", dataset, configs['sklearn'])

    # 8. 随机森林
    model_rf = RandomForestModel(n_estimators=100, max_depth=10)
    results['随机森林'] = run_experiment(model_rf, "随机森林", dataset, configs['sklearn'])

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
