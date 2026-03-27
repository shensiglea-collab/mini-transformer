"""
训练模块 - 统一的训练和评估框架
"""
from .trainer import Trainer, train_model
from .advanced_trainer import AdvancedTrainer
from .metrics import MetricsCalculator

__all__ = ['Trainer', 'train_model', 'AdvancedTrainer', 'MetricsCalculator']
