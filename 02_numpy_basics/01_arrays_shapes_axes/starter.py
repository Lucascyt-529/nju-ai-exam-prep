"""学生练习：NumPy 数组、形状与 axis。"""

from collections.abc import Sequence

import numpy as np


def make_feature_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """将嵌套序列转换为非空、有限的二维浮点数组。"""
    raise NotImplementedError("请完成 make_feature_matrix")


def describe_array(matrix: np.ndarray) -> dict[str, object]:
    """返回 shape、ndim 和 dtype 三项描述。"""
    raise NotImplementedError("请完成 describe_array")


def feature_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个特征的均值，结果形状应为 (n_features,)。"""
    raise NotImplementedError("请完成 feature_means")


def sample_means(matrix: np.ndarray) -> np.ndarray:
    """返回每个样本的均值，结果形状应为 (n_samples,)。"""
    raise NotImplementedError("请完成 sample_means")


def select_feature(matrix: np.ndarray, column: int) -> np.ndarray:
    """选择一个特征列，返回形状为 (n_samples,) 的一维数组。"""
    raise NotImplementedError("请完成 select_feature")
