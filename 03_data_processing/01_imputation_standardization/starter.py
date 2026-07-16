"""学生练习：缺失值填补与标准化。"""

import numpy as np


def fit_mean_imputer(X_train: np.ndarray) -> np.ndarray:
    """只使用训练集计算每个特征的均值填充值。"""
    raise NotImplementedError("请完成 fit_mean_imputer")


def transform_mean_imputer(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """使用给定填充值处理数据，不重新计算统计量。"""
    raise NotImplementedError("请完成 transform_mean_imputer")


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只使用训练集计算均值和安全标准差。"""
    raise NotImplementedError("请完成 fit_standardizer")


def transform_standardizer(
    X: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """使用训练阶段得到的均值和尺度标准化数据。"""
    raise NotImplementedError("请完成 transform_standardizer")
