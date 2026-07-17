"""学生练习：训练集拟合的IQR异常值诊断、裁剪与稳健缩放。"""

import numpy as np


def fit_iqr_bounds(
    X_train: np.ndarray, *, multiplier: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 fit_iqr_bounds")


def detect_outliers(
    X: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 detect_outliers")


def summarize_outliers(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    raise NotImplementedError("请完成 summarize_outliers")


def clip_outliers(
    X: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 clip_outliers")


def fit_robust_scaler(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 fit_robust_scaler")


def transform_robust_scaler(
    X: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_robust_scaler")
