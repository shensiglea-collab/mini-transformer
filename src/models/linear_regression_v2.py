"""
改进版线性回归模型
包含L1/L2正则化和更好的初始化
"""
import torch
import torch.nn as nn


class LinearRegressionL2(nn.Module):
    """
    带L2正则化（Ridge回归）的线性回归
    通过weight_decay参数实现
    """

    def __init__(self, input_size=13, output_size=1, weight_decay=0.01):
        """
        初始化L2正则化线性回归

        Args:
            input_size: 输入特征维度
            output_size: 输出维度
            weight_decay: L2正则化系数
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.weight_decay = weight_decay

        # 更好的初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def get_model_info(self):
        return {
            'name': 'Linear Regression (L2)',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'带L2正则化的线性回归 (weight_decay={self.weight_decay})'
        }


class LinearRegressionL1(nn.Module):
    """
    带L1正则化的线性回归
    使用自定义的L1损失
    """

    def __init__(self, input_size=13, output_size=1, l1_lambda=0.01):
        """
        初始化L1正则化线性回归

        Args:
            input_size: 输入特征维度
            output_size: 输出维度
            l1_lambda: L1正则化系数
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.l1_lambda = l1_lambda

        # 更好的初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def l1_loss(self):
        """计算L1正则化项"""
        return self.l1_lambda * torch.sum(torch.abs(self.linear.weight))

    def get_model_info(self):
        return {
            'name': 'Linear Regression (L1)',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'带L1正则化的线性回归 (L1_lambda={self.l1_lambda})'
        }


class LinearRegressionElasticNet(nn.Module):
    """
    Elastic Net回归（L1 + L2组合）
    """

    def __init__(self, input_size=13, output_size=1, l1_lambda=0.01, l2_lambda=0.01):
        """
        初始化Elastic Net回归

        Args:
            input_size: 输入特征维度
            output_size: 输出维度
            l1_lambda: L1正则化系数
            l2_lambda: L2正则化系数
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # 更好的初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        """计算组合正则化项"""
        l1_loss = self.l1_lambda * torch.sum(torch.abs(self.linear.weight))
        l2_loss = self.l2_lambda * torch.sum(self.linear.weight ** 2)
        return l1_loss + l2_loss

    def get_model_info(self):
        return {
            'name': 'Linear Regression (ElasticNet)',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'Elastic Net (L1={self.l1_lambda}, L2={self.l2_lambda})'
        }


class ImprovedLinearRegression(nn.Module):
    """
    改进版线性回归
    包含：批量归一化、Dropout、更好的初始化
    """

    def __init__(self, input_size=13, output_size=1, dropout=0.1, use_bn=True):
        """
        初始化改进版线性回归

        Args:
            input_size: 输入特征维度
            output_size: 输出维度
            dropout: Dropout概率
            use_bn: 是否使用批量归一化
        """
        super().__init__()
        self.use_bn = use_bn

        self.linear = nn.Linear(input_size, output_size)

        if use_bn:
            self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout)

        # 更好的初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        x = self.dropout(x)
        return self.linear(x)

    def get_model_info(self):
        return {
            'name': 'Improved Linear Regression',
            'params': sum(p.numel() for p in self.parameters()),
            'description': f'带BN和Dropout的线性回归'
        }
