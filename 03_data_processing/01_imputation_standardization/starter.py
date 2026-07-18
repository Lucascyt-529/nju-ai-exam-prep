"""学生练习：缺失值填补与标准化。"""

import numpy as np


def fit_mean_imputer(X_train: np.ndarray) -> np.ndarray:
    """只使用训练集计算每个特征的均值填充值。"""
    return np.nanmean(X_train, axis=0)


def transform_mean_imputer(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """使用给定填充值处理数据，不重新计算统计量。"""
    return np.where(np.isnan(X), fill_values, X)


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只使用训练集计算均值和安全标准差。"""
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1
    return means, stds
def transform_standardizer(
    X: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """使用训练阶段得到的均值和尺度标准化数据。"""
    return (X - means) / scales
