"""
线性回归模型
简单的单层线性模型作为基准
"""
import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    """
    线性回归模型
    作为最简单的基准模型，用于对比神经网络的效果
    """

    def __init__(self, input_size=13, output_size=1):
        """
        初始化线性回归模型

        Args:
            input_size: 输入特征维度 (Boston Housing为13)
            output_size: 输出维度 (房价预测为1)
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_size]

        Returns:
            预测值 [batch_size, output_size]
        """
        return self.linear(x)

    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'Linear Regression',
            'params': sum(p.numel() for p in self.parameters()),
            'description': '简单的线性回归模型'
        }
