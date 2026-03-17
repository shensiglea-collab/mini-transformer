"""
评估指标模块
提供统一的模型评估指标计算
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict


class MetricsCalculator:
    """指标计算器 - 统一计算各种评估指标"""

    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            包含各项指标的字典
        """
        # 确保是一维数组
        y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true
        y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, float], dataset_name: str = ""):
        """
        打印指标

        Args:
            metrics: 指标字典
            dataset_name: 数据集名称（如"训练集"、"测试集"）
        """
        prefix = f"{dataset_name}" if dataset_name else ""
        print(f"\n   {prefix}结果:")
        print(f"     MSE:  {metrics['mse']:.4f}")
        print(f"     RMSE: {metrics['rmse']:.4f}")
        print(f"     MAE:  ${metrics['mae']:.2f}k")
        print(f"     R²:   {metrics['r2']:.4f}")

    @staticmethod
    def print_comparison(metrics_dict: Dict[str, Dict[str, float]]):
        """
        打印多个模型的对比结果

        Args:
            metrics_dict: {模型名: 指标字典} 的字典
        """
        print("\n" + "="*70)
        print("模型对比结果")
        print("="*70)
        print(f"{'模型':<20} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-"*70)

        for model_name, metrics in metrics_dict.items():
            print(f"{model_name:<20} "
                  f"{metrics['mse']:<10.4f} "
                  f"{metrics['rmse']:<10.4f} "
                  f"${metrics['mae']:<9.2f}k "
                  f"{metrics['r2']:<10.4f}")

        print("="*70)


def print_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 5):
    """
    打印预测示例

    Args:
        y_true: 真实值
        y_pred: 预测值
        num_samples: 显示的样本数
    """
    print(f"\n   预测示例 (前{num_samples}个样本):")
    print(f"   {'序号':<4} {'真实价格':<10} {'预测价格':<10} {'误差':<10}")
    print("   " + "-"*40)

    for i in range(min(num_samples, len(y_true))):
        actual = y_true[i]
        predicted = y_pred[i][0] if len(y_pred[i].shape) > 0 else y_pred[i]
        error = abs(actual - predicted)
        print(f"   {i+1:<4} ${actual:<9.2f}k ${predicted:<9.2f}k ${error:<9.2f}k")
