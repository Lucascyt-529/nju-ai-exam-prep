"""参考实现：矩阵乘法与线性输出。"""

import numpy as np


def validate_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} 必须是非空二维数组")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须只包含有限数值")


def matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """返回两个二维矩阵的乘积。"""
    validate_matrix(left, "left")
    validate_matrix(right, "right")
    if left.shape[1] != right.shape[0]:
        raise ValueError("left 的列数必须等于 right 的行数")
    return left @ right


def linear_scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """计算 X @ w + b，返回每个样本一个分数。"""
    validate_matrix(X, "X")
    if w.ndim != 1 or w.shape[0] != X.shape[1]:
        raise ValueError("w 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(w)) or not np.isfinite(b):
        raise ValueError("w 和 b 必须只包含有限数值")
    return X @ w + b


def multi_output_scores(
    X: np.ndarray, W: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """计算 X @ W + b，返回每个样本的多个分数。"""
    validate_matrix(X, "X")
    validate_matrix(W, "W")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X 的特征数必须等于 W 的行数")
    if b.ndim != 1 or b.shape[0] != W.shape[1]:
        raise ValueError("b 必须具有形状 (n_outputs,)")
    if not np.all(np.isfinite(b)):
        raise ValueError("b 必须只包含有限数值")
    return X @ W + b


def feature_gram_matrix(X: np.ndarray) -> np.ndarray:
    """返回 X.T @ X。"""
    validate_matrix(X, "X")
    return X.T @ X
