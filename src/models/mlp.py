"""
MLP (多层感知机) 模型
包含简单MLP和深层MLP两个版本
"""
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    简单MLP模型
    2层隐藏层，适合中等复杂度的房价预测
    """

    def __init__(self, input_size=13, hidden_size=64, output_size=1, dropout=0.2):
        """
        初始化MLP模型

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            dropout: Dropout概率
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_size]

        Returns:
            预测值 [batch_size, output_size]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'MLP (2-layer)',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'2层MLP，隐藏层大小: 64'
        }


class DeepMLPModel(nn.Module):
    """
    深层MLP模型
    可配置的多层隐藏层，适合更复杂的模式学习
    """

    def __init__(self, input_size=13, hidden_sizes=[128, 64, 32],
                 output_size=1, dropout=0.3):
        """
        初始化深层MLP模型

        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度
            dropout: Dropout概率
        """
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_size]

        Returns:
            预测值 [batch_size, output_size]
        """
        return self.model(x)

    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'Deep MLP',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'深层MLP，隐藏层: {self.model}'
        }
