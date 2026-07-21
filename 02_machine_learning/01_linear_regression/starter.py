"""学生练习：按 predict -> MSE -> gradients -> GD -> least squares 实现。

本专题统一使用以下 shape：X:(n_samples, n_features)、y:(n_samples,)、
w:(n_features,)、prediction/error:(n_samples,)。不要把 (n,) 与 (n, 1) 混用。
"""

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """计算线性预测。

    输入：X:(n_samples, n_features)、w:(n_features,)、b: Python float。
    输出：prediction:(n_samples,)。
    """
    raise NotImplementedError("请完成 predict")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均平方误差。

    输入：y_true:(n_samples,)、y_pred:(n_samples,)。
    输出：一个 Python float。
    """
    raise NotImplementedError("请完成 mean_squared_error")


def mse_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """计算 MSE 对 w 和 b 的解析梯度。

    输入：X:(n_samples, n_features)、y:(n_samples,)、w:(n_features,)、b: float。
    输出：gradient_w:(n_features,) 和 gradient_b: Python float。
    """
    raise NotImplementedError("请完成 mse_gradients")


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    initial_w: np.ndarray | None = None,
    initial_b: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """使用批量梯度下降拟合线性回归。

    输入：X:(n_samples, n_features)、y:(n_samples,) 及训练设置。
    输出：w:(n_features,)、b: float、loss_history: 一维数组。
    loss_history 先保存初始损失，因此长度是 n_steps + 1。
    """
    raise NotImplementedError("请完成 fit_gradient_descent")


def fit_least_squares(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """使用 np.linalg.lstsq 拟合线性回归。

    输入：X:(n_samples, n_features)、y:(n_samples,)；fit_intercept 决定是否拟合截距。
    输出：w:(n_features,) 和 b: Python float。
    """
    raise NotImplementedError("请完成 fit_least_squares")
