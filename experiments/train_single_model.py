"""
单模型训练脚本 - 训练指定模型
使用: python experiments/train_single_model.py --model mlp
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_housing_data
from src.models import LinearRegressionModel, MLPModel, DeepMLPModel, TransformerRegressor
from src.training import Trainer
from src.training.metrics import print_predictions


def get_model(model_name, input_size=13):
    """根据名称获取模型实例"""
    models = {
        'linear': LinearRegressionModel(input_size, 1),
        'mlp': MLPModel(input_size, hidden_size=64, output_size=1),
        'deep_mlp': DeepMLPModel(input_size, hidden_sizes=[128, 64, 32], output_size=1),
        'transformer': TransformerRegressor(input_size, d_model=32, num_heads=2, d_ff=64, num_layers=1)
    }

    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可用选项: {list(models.keys())}")

    return models[model_name]


def get_config(model_name):
    """获取模型训练配置"""
    configs = {
        'linear': {'epochs': 100, 'batch_size': 32, 'lr': 0.01},
        'mlp': {'epochs': 100, 'batch_size': 32, 'lr': 0.001},
        'deep_mlp': {'epochs': 150, 'batch_size': 32, 'lr': 0.0005},
        'transformer': {'epochs': 100, 'batch_size': 32, 'lr': 0.001}
    }
    return configs.get(model_name, {'epochs': 100, 'batch_size': 32, 'lr': 0.001})


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练单个房价预测模型')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['linear', 'mlp', 'deep_mlp', 'transformer'],
                        help='选择模型: linear, mlp, deep_mlp, transformer')
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
    print(f"\n📊 模型信息:")
    print(f"   模型类型: {args.model}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    print(f"\n⚙️ 训练配置:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['lr']}")

    # 创建训练器并训练
    trainer = Trainer(model, model_name=args.model, lr=config['lr'])

    print(f"\n🔥 开始训练...")
    trainer.train(
        dataset.X_train_tensor,
        dataset.y_train_tensor,
        num_epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=True
    )

    # 评估
    print(f"\n📈 模型评估:")
    y_train, y_test = dataset.get_numpy_data()
    report = trainer.get_full_report(
        dataset.X_train_tensor, dataset.y_train_tensor,
        dataset.X_test_tensor, dataset.y_test_tensor,
        y_train, y_test
    )

    # 打印结果
    from src.training.metrics import MetricsCalculator
    MetricsCalculator.print_metrics(report['train'], "训练集")
    MetricsCalculator.print_metrics(report['test'], "测试集")

    # 预测示例
    print_predictions(y_test, report['test_pred'], num_samples=10)

    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
