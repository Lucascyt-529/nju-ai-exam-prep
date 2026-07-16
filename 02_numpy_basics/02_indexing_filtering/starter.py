"""学生练习：NumPy 索引、筛选和同步打乱。"""

from collections.abc import Sequence

import numpy as np


def select_rows(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """按给定顺序选择多行，保持二维结果。"""
    raise NotImplementedError("请完成 select_rows")


def filter_rows_by_feature(
    matrix: np.ndarray, column: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """保留指定特征大于等于阈值的行，同时返回布尔掩码。"""
    raise NotImplementedError("请完成 filter_rows_by_feature")


def split_features_target(
    table: np.ndarray, target_column: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """将完整数据表切分为二维特征 X 和一维标签 y。"""
    raise NotImplementedError("请完成 split_features_target")


def shuffle_in_unison(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用同一组下标同步打乱 X 和 y，并返回排列下标。"""
    raise NotImplementedError("请完成 shuffle_in_unison")
