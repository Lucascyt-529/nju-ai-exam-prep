"""学生练习：留出法、分层划分、K折与自助采样。"""

import numpy as np


def train_test_split_indices(
    n_samples: int, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """返回互不重叠的训练和测试索引。"""
    raise NotImplementedError("请完成 train_test_split_indices")


def stratified_train_test_split_indices(
    y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """每个类别分别划分，再合并训练和测试索引。"""
    raise NotImplementedError("请完成 stratified_train_test_split_indices")


def kfold_indices(
    n_samples: int, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回K组训练索引和验证索引。"""
    raise NotImplementedError("请完成 kfold_indices")


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回尽量保持类别比例的K折索引。"""
    raise NotImplementedError("请完成 stratified_kfold_indices")


def bootstrap_indices(
    n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """返回等量有放回训练索引和未抽中的袋外索引。"""
    raise NotImplementedError("请完成 bootstrap_indices")
