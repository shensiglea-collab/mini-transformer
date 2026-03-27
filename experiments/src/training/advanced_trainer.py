"""
高级训练器
支持L1/L2正则化和自定义损失函数
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .metrics import MetricsCalculator


class AdvancedTrainer:
    """
    高级训练器
    支持L1/L2正则化和更灵活的训练策略
    """

    def __init__(self, model: nn.Module, model_name: str = "Model",
                 lr: float = 0.001, weight_decay: float = 0.0,
                 device: str = 'cpu'):
        """
        初始化高级训练器

        Args:
            model: 神经网络模型
            model_name: 模型名称
            lr: 学习率
            weight_decay: L2正则化系数（PyTorch默认）
            device: 训练设备
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.criterion = nn.MSELoss()

        # 优化器（支持weight_decay作为L2正则化）
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.train_losses: List[float] = []
        self.metrics_calculator = MetricsCalculator()

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              num_epochs: int = 100, batch_size: int = 32,
              verbose: bool = True, l1_lambda: float = 0.0) -> List[float]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            num_epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印进度
            l1_lambda: L1正则化系数（如果模型支持）

        Returns:
            训练损失历史
        """
        self.model.train()
        self.train_losses = []

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            # Mini-batch训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # 前向传播
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)

                # 添加模型特定的正则化损失
                if hasattr(self.model, 'l1_loss'):
                    loss = loss + self.model.l1_loss()
                elif hasattr(self.model, 'regularization_loss'):
                    loss = loss + self.model.regularization_loss()
                elif l1_lambda > 0:
                    # 手动添加L1正则化
                    l1_reg = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + l1_lambda * l1_reg

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            self.train_losses.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

        if verbose:
            print(f"   训练完成！最终Loss: {self.train_losses[-1]:.6f}")

        return self.train_losses

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor,
                 y_test_numpy: Optional[torch.Tensor] = None) -> Dict:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试目标（张量）
            y_test_numpy: 测试目标（numpy数组）

        Returns:
            评估指标字典
        """
        self.model.eval()

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()

        if y_test_numpy is None:
            y_test_numpy = y_test.cpu().numpy()

        metrics = self.metrics_calculator.calculate(y_test_numpy, predictions)
        metrics['predictions'] = predictions

        return metrics

    def get_full_report(self, X_train: torch.Tensor, y_train: torch.Tensor,
                        X_test: torch.Tensor, y_test: torch.Tensor,
                        y_train_numpy, y_test_numpy) -> Dict:
        """
        获取完整报告

        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            y_train_numpy, y_test_numpy: numpy格式的目标值

        Returns:
            完整评估报告
        """
        self.model.eval()

        with torch.no_grad():
            train_pred = self.model(X_train.to(self.device)).cpu().numpy()
            test_pred = self.model(X_test.to(self.device)).cpu().numpy()

        report = {
            'train': self.metrics_calculator.calculate(y_train_numpy, train_pred),
            'test': self.metrics_calculator.calculate(y_test_numpy, test_pred),
            'train_pred': train_pred,
            'test_pred': test_pred
        }

        return report
