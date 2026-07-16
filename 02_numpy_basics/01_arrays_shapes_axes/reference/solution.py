"""参考实现：NumPy 数组、形状与 axis。"""

from collections.abc import Sequence

import numpy as np


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix 不能为空")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix 必须只包含有限数值")


def make_feature_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """将嵌套序列转换为非空、有限的二维浮点数组。"""
    try:
        matrix = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rows 必须是列数一致的数值行") from exc
    validate_matrix(matrix)
    return matrix


def describe_array(matrix: np.ndarray) -> dict[str, object]:
    """返回 shape、ndim 和 dtype 三项描述。"""
    validate_matrix(matrix)
    return {
        "shape": matrix.shape,
        "ndim": matrix.ndim,
        "dtype": str(matrix.dtype),
    }


def feature_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个特征的均值，结果形状为 (n_features,)。"""
    validate_matrix(matrix)
    return matrix.mean(axis=0)


def sample_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个样本的均值，结果形状为 (n_samples,)。"""
    validate_matrix(matrix)
    return matrix.mean(axis=1)


def select_feature(matrix: np.ndarray, column: int) -> np.ndarray:
    """选择一个特征列，返回形状为 (n_samples,) 的一维数组。"""
    validate_matrix(matrix)
    if not -matrix.shape[1] <= column < matrix.shape[1]:
        raise IndexError("column 超出特征范围")
    return matrix[:, column]
