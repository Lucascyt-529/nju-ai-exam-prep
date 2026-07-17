"""学生练习：NumPy 数组、形状与 axis。"""

from collections.abc import Sequence

import numpy as np


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("数据维数不对")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("样本特征结构错误")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("数值不合法")


def make_feature_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """将嵌套序列转换为非空、有限的二维浮点数组。"""
    matrix = np.asarray(rows, dtype=float)
    _validate_matrix(matrix)
    return matrix


def describe_array(matrix: np.ndarray) -> dict[str, object]:
    """返回 shape、ndim 和 dtype 三项描述。"""
    _validate_matrix(matrix)
    shape_m = matrix.shape
    ndim = matrix.ndim
    dtype = matrix.dtype
    return {
        "shape": shape_m,
        "ndim": ndim,
        "dtype": str(dtype),
    }


def feature_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个特征的均值，结果形状应为 (n_features,)。"""
    return matrix.mean(axis=0)


def sample_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个样本的均值，结果形状应为 (n_samples,)。"""
    return matrix.mean(axis=1)


def select_feature(matrix: np.ndarray, column: int) -> np.ndarray:
    """选择一个特征列，返回形状为 (n_samples,) 的一维数组。"""
    return matrix[:, column]
