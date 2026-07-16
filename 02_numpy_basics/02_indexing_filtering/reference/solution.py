"""参考实现：NumPy 索引、筛选和同步打乱。"""

from collections.abc import Sequence

import numpy as np


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix 必须是非空二维数组")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix 必须只包含有限数值")


def select_rows(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """按给定顺序选择多行，保持二维结果。"""
    validate_matrix(matrix)
    index_array = np.asarray(indices, dtype=int)
    if index_array.ndim != 1:
        raise ValueError("indices 必须是一维")
    return matrix[index_array]


def filter_rows_by_feature(
    matrix: np.ndarray, column: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """保留指定特征大于等于阈值的行，同时返回布尔掩码。"""
    validate_matrix(matrix)
    if not -matrix.shape[1] <= column < matrix.shape[1]:
        raise IndexError("column 超出特征范围")
    mask = matrix[:, column] >= threshold
    return matrix[mask], mask


def split_features_target(
    table: np.ndarray, target_column: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """将完整数据表切分为二维特征 X 和一维标签 y。"""
    validate_matrix(table)
    if table.shape[1] < 2:
        raise ValueError("数据表至少需要一个特征列和一个目标列")
    if not -table.shape[1] <= target_column < table.shape[1]:
        raise IndexError("target_column 超出列范围")

    normalized_column = target_column % table.shape[1]
    y = table[:, normalized_column].copy()
    X = np.delete(table, normalized_column, axis=1)
    return X, y


def shuffle_in_unison(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用同一组下标同步打乱 X 和 y，并返回排列下标。"""
    validate_matrix(X)
    if y.ndim != 1:
        raise ValueError("y 必须是一维数组")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 和 y 的样本数必须一致")

    permutation = np.random.default_rng(seed).permutation(X.shape[0])
    return X[permutation], y[permutation], permutation
