"""学生练习：NumPy 数组创建与 dtype。"""

from collections.abc import Sequence

import numpy as np


def make_float_vector(values: Sequence[float]) -> np.ndarray:
    """把非空一维数值序列转换为浮点数组。"""
    raise NotImplementedError("请完成 make_float_vector")


def make_zero_matrix(n_rows: int, n_columns: int) -> np.ndarray:
    """创建形状为 (n_rows, n_columns) 的全0浮点数组。"""
    raise NotImplementedError("请完成 make_zero_matrix")


def make_step_sequence(start: float, stop: float, step: float) -> np.ndarray:
    """创建左闭右开的浮点等差序列。"""
    raise NotImplementedError("请完成 make_step_sequence")


def describe_array(array: np.ndarray) -> dict[str, object]:
    """返回 shape、ndim、size 和 dtype。"""
    raise NotImplementedError("请完成 describe_array")


def convert_to_float(array: np.ndarray) -> np.ndarray:
    """返回独立的浮点副本，不修改原数组。"""
    raise NotImplementedError("请完成 convert_to_float")
