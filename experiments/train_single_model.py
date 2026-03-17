"""
单模型训练脚本 - 训练指定模型
使用: python experiments/train_single_model.py --model mlp

支持的模型:
- 线性回归系列: linear, linear_l2, linear_l1, linear_elastic, linear_improved
- 神经网络: mlp, deep_mlp, transformer
- 传统机器学习: decision_tree, svm, knn, random_forest
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_housing_data
from src.models import (
    LinearRegressionModel, MLPModel, DeepMLPModel, TransformerRegressor,
    DecisionTreeModel, SVMModel, KNNModel, RandomForestModel,
    LinearRegressionL2, LinearRegressionL1, LinearRegressionElasticNet, ImprovedLinearRegression
)
from src.training import Trainer
from src.training.metrics import print_predictions


def get_model(model_name, input_size=13):
    """根据名称获取模型实例"""
    models = {
        # 线性回归系列
        'linear': LinearRegressionModel(input_size, 1),
        'linear_l2': LinearRegressionL2(input_size, 1, weight_decay=0.01),
        'linear_l1': LinearRegressionL1(input_size, 1, l1_lambda=0.01),
        'linear_elastic': LinearRegressionElasticNet(input_size, 1, l1_lambda=0.01, l2_lambda=0.01),
        'linear_improved': ImprovedLinearRegression(input_size, 1, dropout=0.1, use_bn=True),
        
        # 神经网络模型
        'mlp': MLPModel(input_size, hidden_size=64, output_size=1),
        'deep_mlp': DeepMLPModel(input_size, hidden_sizes=[128, 64, 32], output_size=1),
        'transformer': TransformerRegressor(input_size, d_model=32, num_heads=2, d_ff=64, num_layers=1),
        
        # 传统机器学习模型
        'decision_tree': DecisionTreeModel(max_depth=10),
        'svm': SVMModel(kernel='rbf', C=1.0),
        'knn': KNNModel(n_neighbors=5),
        'random_forest': RandomForestModel(n_estimators=100, max_depth=10)
    }

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(f"未知模型: {model_name}. 可用选项: {available}")

    return models[model_name]


def get_config(model_name):
    """获取模型训练配置"""
    configs = {
        # 线性回归系列
        'linear': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        'linear_l2': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        'linear_l1': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        'linear_elastic': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        'linear_improved': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        
        # 神经网络模型
        'mlp': {'epochs': 100, 'batch_size': 32, 'lr': 0.001},
        'deep_mlp': {'epochs': 150, 'batch_size': 32, 'lr': 0.0005},
        'transformer': {'epochs': 100, 'batch_size': 32, 'lr': 0.001},
        
        # 传统机器学习模型 (不需要这些参数)
        'decision_tree': {'epochs': None, 'batch_size': None, 'lr': None},
        'svm': {'epochs': None, 'batch_size': None, 'lr': None},
        'knn': {'epochs': None, 'batch_size': None, 'lr': None},
        'random_forest': {'epochs': None, 'batch_size': None, 'lr': None}
    }
    return configs.get(model_name, {'epochs': 100, 'batch_size': 32, 'lr': 0.001})


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练单个房价预测模型')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['linear', 'linear_l2', 'linear_l1', 'linear_elastic', 'linear_improved',
                                'mlp', 'deep_mlp', 'transformer',
                                'decision_tree', 'svm', 'knn', 'random_forest'],
                        help='选择模型')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认使用模型预设值)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小 (默认使用模型预设值)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认使用模型预设值)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"🏠 房价预测 - 单模型训练")
    print(f"   模型: {args.model}")
    print("="*70)

    # 加载数据
    print("\n📦 加载数据...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'housing.data.txt')
    dataset = load_housing_data(data_path)
    dataset.print_info()

    # 获取模型和配置
    model = get_model(args.model)
    config = get_config(args.model)

    # 覆盖配置（如果提供了命令行参数）
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr

    # 模型信息
    model_info = model.get_model_info()
    print(f"\n📊 模型信息:")
    if model_info.get('model_type') == 'sklearn':
        print(f"   类型: sklearn")
        print(f"   模型: {model_info['model_name']}")
    else:
        print(f"   模型类型: {args.model}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    if config['epochs'] is not None:
        print(f"\n⚙️ 训练配置:")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Learning Rate: {config['lr']}")
    else:
        print(f"\n⚙️ sklearn模型 (无需训练配置)")

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
        trainer = Trainer(model, model_name=args.model, lr=config['lr'])

        print(f"\n🔥 开始训练...")
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
    from src.training.metrics import MetricsCalculator
    train_metrics = MetricsCalculator.calculate(y_train, train_pred)
    test_metrics = MetricsCalculator.calculate(y_test, test_pred)
    
    MetricsCalculator.print_metrics(train_metrics, "训练集")
    MetricsCalculator.print_metrics(test_metrics, "测试集")

    # 预测示例
    print_predictions(y_test, test_pred, num_samples=10)

    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
