"""参考实现：训练集拟合的IQR异常值诊断、裁剪与稳健缩放。"""

import numpy as np


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and not np.issubdtype(value.dtype, np.bool_)
        and np.all(np.isfinite(value))
    )


def _is_positive_finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )


def _validate_matrix(X: np.ndarray) -> None:
    if (
        not _is_finite_numeric_array(X)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
    ):
        raise ValueError("X必须是非空有限数值二维NumPy数组")


def _validate_feature_vector(
    X: np.ndarray, values: np.ndarray, name: str
) -> None:
    if (
        not _is_finite_numeric_array(values)
        or values.shape != (X.shape[1],)
    ):
        raise ValueError(f"{name}必须是形状为(n_features,)的有限数值数组")


def _validate_bounds(
    X: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray
) -> None:
    _validate_feature_vector(X, lower_bounds, "lower_bounds")
    _validate_feature_vector(X, upper_bounds, "upper_bounds")
    if np.any(lower_bounds > upper_bounds):
        raise ValueError("每个特征的lower_bounds都不能大于upper_bounds")


def fit_iqr_bounds(
    X_train: np.ndarray, *, multiplier: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """只用训练集按列拟合Tukey IQR上下界。"""
    _validate_matrix(X_train)
    if not _is_positive_finite_scalar(multiplier):
        raise ValueError("multiplier必须是正有限数值")

    X_float = X_train.astype(float, copy=False)
    q1 = np.quantile(X_float, 0.25, axis=0)
    q3 = np.quantile(X_float, 0.75, axis=0)
    iqr = q3 - q1
    width = float(multiplier) * iqr
    return q1 - width, q3 + width


def detect_outliers(
    X: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """使用已经拟合的边界逐元素标记异常值，不重新拟合。"""
    _validate_matrix(X)
    _validate_bounds(X, lower_bounds, upper_bounds)
    return (X < lower_bounds) | (X > upper_bounds)


def summarize_outliers(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """返回逐特征异常数、逐样本是否含异常、含异常样本总数。"""
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.ndim != 2
        or mask.shape[0] == 0
        or mask.shape[1] == 0
    ):
        raise ValueError("mask必须是非空二维布尔NumPy数组")
    feature_counts = np.sum(mask, axis=0)
    row_has_outlier = np.any(mask, axis=1)
    return feature_counts, row_has_outlier, int(np.sum(row_has_outlier))


def clip_outliers(
    X: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """使用训练边界逐元素裁剪，并返回新数组。"""
    _validate_matrix(X)
    _validate_bounds(X, lower_bounds, upper_bounds)
    return np.clip(X.astype(float, copy=True), lower_bounds, upper_bounds)


def fit_robust_scaler(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """只用训练集拟合每列中位数和安全IQR尺度。"""
    _validate_matrix(X_train)
    X_float = X_train.astype(float, copy=False)
    medians = np.median(X_float, axis=0)
    q1 = np.quantile(X_float, 0.25, axis=0)
    q3 = np.quantile(X_float, 0.75, axis=0)
    scales = q3 - q1
    scales[scales == 0.0] = 1.0
    return medians, scales


def transform_robust_scaler(
    X: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """使用训练阶段保存的中位数和IQR缩放数据。"""
    _validate_matrix(X)
    _validate_feature_vector(X, medians, "medians")
    _validate_feature_vector(X, scales, "scales")
    if np.any(scales <= 0):
        raise ValueError("scales必须全部大于0")
    return (X.astype(float, copy=True) - medians) / scales
