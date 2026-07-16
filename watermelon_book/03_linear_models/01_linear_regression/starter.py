"""学生练习：NumPy线性回归的预测、损失、梯度与拟合。"""

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """返回形状为 (n_samples,) 的线性预测。"""
    raise NotImplementedError("请完成 predict")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("请完成 mean_squared_error")


def mse_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """返回MSE对w和b的梯度。"""
    raise NotImplementedError("请完成 mse_gradients")


def fit_least_squares(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """使用 np.linalg.lstsq 拟合线性回归。"""
    raise NotImplementedError("请完成 fit_least_squares")


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    initial_w: np.ndarray | None = None,
    initial_b: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """返回w、b和包含初始损失的长度 n_steps+1 损失历史。"""
    raise NotImplementedError("请完成 fit_gradient_descent")
