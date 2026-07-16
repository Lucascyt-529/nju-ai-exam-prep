"""参考实现：缺失值填补与标准化。"""

import numpy as np


def validate_matrix(X: np.ndarray, *, allow_nan: bool) -> None:
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if np.any(np.isinf(X)):
        raise ValueError("X 不能包含无穷大")
    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError("X 不能包含 NaN")


def validate_feature_parameters(X: np.ndarray, values: np.ndarray, name: str) -> None:
    if values.ndim != 1 or values.shape[0] != X.shape[1]:
        raise ValueError(f"{name} 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 必须只包含有限数值")


def fit_mean_imputer(X_train: np.ndarray) -> np.ndarray:
    """只使用训练集计算每个特征的均值填充值。"""
    validate_matrix(X_train, allow_nan=True)
    all_missing = np.all(np.isnan(X_train), axis=0)
    if np.any(all_missing):
        columns = np.flatnonzero(all_missing).tolist()
        raise ValueError(f"以下特征全部缺失，无法计算均值: {columns}")
    return np.nanmean(X_train, axis=0)


def transform_mean_imputer(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """使用给定填充值处理数据，不重新计算统计量。"""
    validate_matrix(X, allow_nan=True)
    validate_feature_parameters(X, fill_values, "fill_values")
    return np.where(np.isnan(X), fill_values, X)


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只使用训练集计算均值和安全标准差。"""
    validate_matrix(X_train, allow_nan=False)
    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales[scales == 0] = 1.0
    return means, scales


def transform_standardizer(
    X: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """使用训练阶段得到的均值和尺度标准化数据。"""
    validate_matrix(X, allow_nan=False)
    validate_feature_parameters(X, means, "means")
    validate_feature_parameters(X, scales, "scales")
    if np.any(scales <= 0):
        raise ValueError("scales 必须全部大于0")
    return (X - means) / scales
