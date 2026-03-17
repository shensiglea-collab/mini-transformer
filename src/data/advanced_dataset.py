"""
高级数据集处理模块
包含更好的特征工程和预处理选项
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class AdvancedHousingDataset:
    """
    高级房价数据集类
    支持多种特征工程选项和更好的预处理
    """

    # 特征名称
    FEATURE_NAMES = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # 量纲差异大的特征（需要特殊处理）
    HIGH_SCALE_FEATURES = ['CRIM', 'ZN', 'TAX', 'B']

    def __init__(self, data_path, test_size=0.2, random_state=42,
                 exclude_b=False, scaler_type='robust',
                 add_polynomial=False, degree=2):
        """
        初始化高级数据集

        Args:
            data_path: 数据文件路径
            test_size: 测试集比例
            random_state: 随机种子
            exclude_b: 是否排除B字段（黑人比例指数，可能已被弃用）
            scaler_type: 标准化方法 ('standard', 'robust', 'minmax')
            add_polynomial: 是否添加多项式特征
            degree: 多项式次数
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.exclude_b = exclude_b
        self.scaler_type = scaler_type
        self.add_polynomial = add_polynomial
        self.degree = degree

        # 数据存储
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tensor = None
        self.y_train_tensor = None
        self.X_test_tensor = None
        self.y_test_tensor = None
        self.scaler = None
        self.feature_names = self.FEATURE_NAMES.copy()

        # 加载和处理数据
        self._load_data()
        self._feature_engineering()
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
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def _feature_engineering(self):
        """特征工程"""
        # 排除B字段（索引11）
        if self.exclude_b:
            b_index = self.FEATURE_NAMES.index('B')
            self.X = np.delete(self.X, b_index, axis=1)
            self.feature_names = [f for f in self.feature_names if f != 'B']
            print(f"   已排除B字段，剩余特征数: {len(self.feature_names)}")

        # 对量纲差异大的特征进行对数变换（处理长尾分布）
        for feature in ['CRIM', 'TAX']:
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                # 加1避免log(0)
                self.X[:, idx] = np.log1p(self.X[:, idx])

        # 添加多项式特征（可选）
        if self.add_polynomial:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            self.X = poly.fit_transform(self.X)
            print(f"   添加多项式特征后，特征数: {self.X.shape[1]}")

    def _split_and_scale(self):
        """分割数据并标准化"""
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # 选择标准化器
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            # RobustScaler对异常值更鲁棒
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # 标准化
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
        """获取numpy格式的数据"""
        return self.y_train, self.y_test

    def print_info(self):
        """打印数据集信息"""
        print(f"   数据集大小: {self.X.shape}")
        print(f"   特征数: {self.X.shape[1]}")
        print(f"   样本数: {self.X.shape[0]}")
        print(f"   房价范围: ${self.y.min():.2f}k - ${self.y.max():.2f}k")
        print(f"   房价平均值: ${self.y.mean():.2f}k")
        print(f"   训练集: {self.X_train.shape[0]} 样本")
        print(f"   测试集: {self.X_test.shape[0]} 样本")
        print(f"   标准化方法: {self.scaler_type}")
        if self.exclude_b:
            print(f"   已排除B字段")


def load_advanced_housing_data(data_path, exclude_b=True, scaler_type='robust', **kwargs):
    """
    便捷的工厂函数 - 创建高级数据集实例

    Args:
        data_path: 数据文件路径
        exclude_b: 是否排除B字段
        scaler_type: 标准化方法
        **kwargs: 其他参数

    Returns:
        AdvancedHousingDataset实例
    """
    return AdvancedHousingDataset(
        data_path,
        exclude_b=exclude_b,
        scaler_type=scaler_type,
        **kwargs
    )
