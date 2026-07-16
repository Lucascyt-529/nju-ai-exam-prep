"""学生练习：NumPy 广播与安全标准化。"""

import numpy as np


def add_feature_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个特征加一个偏移量。"""
    raise NotImplementedError("请完成 add_feature_offsets")


def add_sample_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个样本的全部特征加同一个偏移量。"""
    raise NotImplementedError("请完成 add_sample_offsets")


def center_features(matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    """使用每个特征的均值进行中心化。"""
    raise NotImplementedError("请完成 center_features")


def safe_standardize(
    matrix: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """使用给定统计量标准化，并返回实际使用的安全标准差。"""
    raise NotImplementedError("请完成 safe_standardize")
