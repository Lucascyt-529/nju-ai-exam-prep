"""参考实现：NumPy 数组创建与 dtype。"""

from collections.abc import Sequence

import numpy as np


def make_float_vector(values: Sequence[float]) -> np.ndarray:
    """把非空一维数值序列转换为浮点数组。"""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values 必须是一维数值序列") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values 必须是非空一维序列")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values 必须只包含有限数值")
    return vector


def make_zero_matrix(n_rows: int, n_columns: int) -> np.ndarray:
    """创建形状为 (n_rows, n_columns) 的全0浮点数组。"""
    if not isinstance(n_rows, int) or not isinstance(n_columns, int):
        raise TypeError("行数和列数必须是整数")
    if n_rows <= 0 or n_columns <= 0:
        raise ValueError("行数和列数必须为正数")
    return np.zeros((n_rows, n_columns), dtype=float)


def make_step_sequence(start: float, stop: float, step: float) -> np.ndarray:
    """创建左闭右开的浮点等差序列。"""
    if not np.all(np.isfinite([start, stop, step])):
        raise ValueError("start、stop 和 step 必须是有限数值")
    if step == 0:
        raise ValueError("step 不能为0")
    return np.arange(start, stop, step, dtype=float)


def describe_array(array: np.ndarray) -> dict[str, object]:
    """返回 shape、ndim、size 和 dtype。"""
    if not isinstance(array, np.ndarray):
        raise TypeError("array 必须是 NumPy 数组")
    return {
        "shape": array.shape,
        "ndim": array.ndim,
        "size": array.size,
        "dtype": str(array.dtype),
    }


def convert_to_float(array: np.ndarray) -> np.ndarray:
    """返回独立的浮点副本，不修改原数组。"""
    if not isinstance(array, np.ndarray):
        raise TypeError("array 必须是 NumPy 数组")
    try:
        converted = array.astype(float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("array 必须能够转换为浮点数") from exc
    if not np.all(np.isfinite(converted)):
        raise ValueError("转换结果必须只包含有限数值")
    return converted
