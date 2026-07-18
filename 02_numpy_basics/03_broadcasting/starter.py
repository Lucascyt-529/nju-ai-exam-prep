"""学生练习：NumPy 广播与安全标准化。"""

import numpy as np


def add_feature_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个特征加一个偏移量。"""
    if offsets.ndim != 1 or matrix.shape[1] != offsets.shape[0]:
        raise ValueError("输入的不是偏移量")
    return matrix + offsets


def add_sample_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个样本的全部特征加同一个偏移量。"""
    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        raise ValueError("输入的长度不符合")
    return matrix + offsets[:, None]


def center_features(matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    """使用每个特征的均值进行中心化。"""
    if means.ndim != 1 or matrix.shape[1] != means.shape[0]:
        raise ValueError("维度不匹配")
    return matrix - means


def safe_standardize(
    matrix: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """使用给定统计量标准化，并返回实际使用的安全标准差。"""
    if (
        means.ndim != 1
        or matrix.shape[1] != means.shape[0]
        or stds.ndim != 1
        or matrix.shape[1] != stds.shape[0]
    ):
        raise ValueError("维度不匹配")

    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0

    return (matrix - means) / stds_safe, stds_safe
