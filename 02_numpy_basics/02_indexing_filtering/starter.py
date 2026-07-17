"""学生练习：NumPy 索引、筛选和同步打乱。"""

from collections.abc import Sequence

import numpy as np


def select_rows(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """按给定顺序选择多行，保持二维结果。"""
    row_indices = np.asarray(indices, dtype=int)
    return matrix[row_indices]


def filter_rows_by_feature(
    matrix: np.ndarray, column: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """保留指定特征大于等于阈值的行，同时返回布尔掩码。"""
    mask = matrix[:, column] >= threshold

    return matrix[mask], mask


def split_features_target(
    table: np.ndarray, target_column: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """将完整数据表切分为二维特征 X 和一维标签 y。"""
    labels = table[:, target_column]
    features = np.delete(table, target_column, axis=1)
    return features, labels


def shuffle_in_unison(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用同一组下标同步打乱 X 和 y，并返回排列下标。"""
    if X.shape[0] != y.shape[0]:
        raise ValueError("行数不一样")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(X.shape[0])
    return X[permutation], y[permutation], permutation
