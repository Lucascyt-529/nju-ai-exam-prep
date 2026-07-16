"""学生练习：明确改变 NumPy 数组形状。"""

import numpy as np


def as_column(vector: np.ndarray) -> np.ndarray:
    """把非空一维数组转换为 (n, 1)。"""
    raise NotImplementedError("请完成 as_column")


def as_flat(column: np.ndarray) -> np.ndarray:
    """把非空 (n, 1) 列数组转换为 (n,)。"""
    raise NotImplementedError("请完成 as_flat")


def transpose_features(X: np.ndarray) -> np.ndarray:
    """把 (n_samples, n_features) 转为 (n_features, n_samples)。"""
    raise NotImplementedError("请完成 transpose_features")


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """在 X 左侧添加一列1。"""
    raise NotImplementedError("请完成 add_bias_column")


def stack_sample_batches(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """沿样本轴拼接两个特征数相同的二维数组。"""
    raise NotImplementedError("请完成 stack_sample_batches")
