"""学生练习：矩阵乘法与线性输出。"""

import numpy as np


def matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """返回两个二维矩阵的乘积。"""
    raise NotImplementedError("请完成 matrix_product")


def linear_scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """计算 X @ w + b，返回每个样本一个分数。"""
    raise NotImplementedError("请完成 linear_scores")


def multi_output_scores(
    X: np.ndarray, W: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """计算 X @ W + b，返回每个样本的多个分数。"""
    raise NotImplementedError("请完成 multi_output_scores")


def feature_gram_matrix(X: np.ndarray) -> np.ndarray:
    """返回 X.T @ X。"""
    raise NotImplementedError("请完成 feature_gram_matrix")
