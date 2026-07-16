"""参考实现：NumPy 广播与安全标准化。"""

import numpy as np


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix 必须是非空二维数组")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix 必须只包含有限数值")


def validate_feature_vector(
    matrix: np.ndarray, vector: np.ndarray, name: str
) -> None:
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须只包含有限数值")


def add_feature_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个特征加一个偏移量。"""
    validate_matrix(matrix)
    validate_feature_vector(matrix, offsets, "offsets")
    return matrix + offsets


def add_sample_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """为每个样本的全部特征加同一个偏移量。"""
    validate_matrix(matrix)
    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        raise ValueError("offsets 必须具有形状 (n_samples,)")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("offsets 必须只包含有限数值")
    return matrix + offsets[:, None]


def center_features(matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    """使用每个特征的均值进行中心化。"""
    validate_matrix(matrix)
    validate_feature_vector(matrix, means, "means")
    return matrix - means


def safe_standardize(
    matrix: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """使用给定统计量标准化，并返回实际使用的安全标准差。"""
    validate_matrix(matrix)
    validate_feature_vector(matrix, means, "means")
    validate_feature_vector(matrix, stds, "stds")
    if np.any(stds < 0):
        raise ValueError("stds 不能为负数")

    safe_stds = stds.copy()
    safe_stds[safe_stds == 0] = 1.0
    return (matrix - means) / safe_stds, safe_stds
