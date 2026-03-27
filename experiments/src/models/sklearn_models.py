"""
传统机器学习模型 - 使用scikit-learn
包含决策树、支持向量机、K近邻等模型
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class SklearnModelWrapper:
    """
    sklearn模型包装器，使其与PyTorch模型接口兼容
    """

    def __init__(self, model):
        """
        初始化包装器

        Args:
            model: sklearn模型实例
        """
        self.model = model
        self.model_name = model.__class__.__name__

    def fit(self, X, y):
        """
        训练模型

        Args:
            X: 训练特征 [n_samples, n_features]
            y: 训练标签 [n_samples]
        """
        # 转换为numpy数组
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if hasattr(y, 'numpy'):
            y = y.numpy()

        self.model.fit(X, y)

    def predict(self, X):
        """
        预测

        Args:
            X: 输入特征 [n_samples, n_features]

        Returns:
            预测值 [n_samples]
        """
        if hasattr(X, 'numpy'):
            X = X.numpy()
        return self.model.predict(X)

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'sklearn',
            'model_name': self.model_name,
            'parameters': self.model.get_params()
        }

    def parameters(self):
        """模拟PyTorch的parameters()方法，返回空列表"""
        return []


class DecisionTreeModel(SklearnModelWrapper):
    """
    决策树回归模型
    """

    def __init__(self, max_depth=None, random_state=42):
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        super().__init__(model)


class SVMModel(SklearnModelWrapper):
    """
    支持向量机回归模型
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        super().__init__(model)


class KNNModel(SklearnModelWrapper):
    """
    K近邻回归模型
    """

    def __init__(self, n_neighbors=5, weights='uniform'):
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        super().__init__(model)


class RandomForestModel(SklearnModelWrapper):
    """
    随机森林回归模型
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        super().__init__(model)