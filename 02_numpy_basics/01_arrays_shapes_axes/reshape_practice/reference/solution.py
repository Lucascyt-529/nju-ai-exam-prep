"""参考实现：明确改变 NumPy 数组形状。"""

import numpy as np


def _validate_matrix(matrix: np.ndarray, name: str) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} 必须是 NumPy 数组")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} 必须是非空二维数组")


def as_column(vector: np.ndarray) -> np.ndarray:
    """把非空一维数组转换为 (n, 1)。"""
    if not isinstance(vector, np.ndarray):
        raise TypeError("vector 必须是 NumPy 数组")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector 必须是非空一维数组")
    return vector.reshape(-1, 1)


def as_flat(column: np.ndarray) -> np.ndarray:
    """把非空 (n, 1) 列数组转换为 (n,)。"""
    _validate_matrix(column, "column")
    if column.shape[1] != 1:
        raise ValueError("column 必须具有形状 (n, 1)")
    return column.reshape(-1)


def transpose_features(X: np.ndarray) -> np.ndarray:
    """把 (n_samples, n_features) 转为 (n_features, n_samples)。"""
    _validate_matrix(X, "X")
    return X.T


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """在 X 左侧添加一列1。"""
    _validate_matrix(X, "X")
    return np.column_stack((np.ones(X.shape[0], dtype=float), X))


def stack_sample_batches(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """沿样本轴拼接两个特征数相同的二维数组。"""
    _validate_matrix(first, "first")
    _validate_matrix(second, "second")
    if first.shape[1] != second.shape[1]:
        raise ValueError("两个批次的特征数必须相同")
    return np.concatenate((first, second), axis=0)
