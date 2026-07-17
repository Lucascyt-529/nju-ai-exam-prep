"""学生练习：明确改变 NumPy 数组形状。"""

import numpy as np


def _validate_matrix(matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("不是numpy数组")
    if matrix.ndim != 2:
        raise ValueError("数据维数不对")
    if matrix.size == 0:
        raise ValueError("数组是空的")


def as_column(vector: np.ndarray) -> np.ndarray:
    """把非空一维数组转换为 (n, 1)。"""
    if not isinstance(vector, np.ndarray):
        raise TypeError("不是向量")

    if vector.ndim != 1:
        raise ValueError("不是一维数组")
    if vector.size == 0:
        raise ValueError("输入的是空数组")

    return vector.reshape(-1, 1)


def as_flat(column: np.ndarray) -> np.ndarray:
    """把非空 (n, 1) 列数组转换为 (n,)。"""
    _validate_matrix(column)
    if column.shape[1] != 1:
        raise ValueError("不是(n,1)类型")

    return column.reshape(-1)


def transpose_features(X: np.ndarray) -> np.ndarray:
    """把 (n_samples, n_features) 转为 (n_features, n_samples)。"""
    _validate_matrix(X)
    return X.T


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """在 X 左侧添加一列1。"""
    _validate_matrix(X)
    n_samples = X.shape[0]
    tmp = np.ones((n_samples, 1))
    return np.concatenate((tmp, X), axis=1)


def stack_sample_batches(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """沿样本轴拼接两个特征数相同的二维数组。"""
    _validate_matrix(first)
    _validate_matrix(second)

    if first.shape[1] != second.shape[1]:
        raise ValueError("列数不对齐")
    return np.concatenate((first, second), axis=0)
