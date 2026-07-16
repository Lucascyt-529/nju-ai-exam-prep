"""学生练习：多项式复杂度、学习曲线与偏差—方差。"""

from collections.abc import Sequence

import numpy as np


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """返回从 x^0 到 x^degree 的二维设计矩阵。"""
    raise NotImplementedError("请完成 polynomial_features")


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """返回按升幂排列的最小二乘系数。"""
    raise NotImplementedError("请完成 fit_polynomial")


def predict_polynomial(
    x: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_polynomial")


def complexity_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    degrees: Sequence[int],
) -> dict[str, np.ndarray]:
    """返回 degrees、train_mse 和 validation_mse。"""
    raise NotImplementedError("请完成 complexity_curve")


def learning_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    degree: int,
    train_sizes: Sequence[int],
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 learning_curve")


def simulate_polynomial_predictions(
    degree: int,
    n_train: int,
    n_repeats: int,
    x_evaluation: np.ndarray,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在 sin(pi*x) 上重复采样，返回预测矩阵与真实值。"""
    raise NotImplementedError("请完成 simulate_polynomial_predictions")


def bias_variance_components(
    predictions: np.ndarray,
    true_values: np.ndarray,
    noise_variance: float,
) -> dict[str, float | np.ndarray]:
    raise NotImplementedError("请完成 bias_variance_components")
