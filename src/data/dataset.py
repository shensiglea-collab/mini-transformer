"""
房价数据集处理模块
提供统一的数据加载和预处理接口
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousingDataset:
    """房价数据集类 - 统一的数据管理"""

    # Boston Housing数据集的特征名称
    FEATURE_NAMES = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # 原始数据
        self.X = None
        self.y = None

        # 分割后的数据
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 张量格式数据
        self.X_train_tensor = None
        self.y_train_tensor = None
        self.X_test_tensor = None
        self.y_test_tensor = None

        # 标准化器
        self.scaler = None

        # 加载数据
        self._load_data()
        self._split_and_scale()

    def _load_data(self):
        """从文件加载原始数据"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('###'):
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 14:  # 13 features + 1 target
                        data.append(values)
                except ValueError:
                    continue

        data = np.array(data)
        self.X = data[:, :-1]  # 前13列是特征
        self.y = data[:, -1]   # 最后一列是目标（价格）

    def _split_and_scale(self):
        """分割数据并标准化"""
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # 转换为张量
        self.X_train_tensor = torch.FloatTensor(X_train_scaled)
        self.y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)
        self.X_test_tensor = torch.FloatTensor(X_test_scaled)
        self.y_test_tensor = torch.FloatTensor(self.y_test).unsqueeze(1)

    def get_train_data(self):
        """获取训练数据"""
        return self.X_train_tensor, self.y_train_tensor

    def get_test_data(self):
        """获取测试数据"""
        return self.X_test_tensor, self.y_test_tensor

    def get_numpy_data(self):
        """获取numpy格式的数据（用于评估）"""
        return self.y_train, self.y_test

    def print_info(self):
        """打印数据集信息"""
        print(f"   数据集大小: {self.X.shape}")
        print(f"   特征数: {self.X.shape[1]}")
        print(f"   样本数: {self.X.shape[0]}")
        print(f"   房价范围: ${self.y.min():.2f}k - ${self.y.max():.2f}k (单位: 千美元)")
        print(f"   房价平均值: ${self.y.mean():.2f}k")
        print(f"   训练集: {self.X_train.shape[0]} 样本")
        print(f"   测试集: {self.X_test.shape[0]} 样本")


def load_housing_data(data_path, test_size=0.2, random_state=42):
    """
    便捷的工厂函数 - 创建数据集实例

    Args:
        data_path: 数据文件路径
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        HousingDataset实例
    """
    return HousingDataset(data_path, test_size, random_state)
