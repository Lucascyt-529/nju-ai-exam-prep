"""学生练习：NumPy线性回归的预测、损失、梯度与拟合。"""

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """返回形状为 (n_samples,) 的线性预测。"""
    return X @ w + b


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y_true
    squared_error = error ** 2
    return squared_error.mean()


def mse_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """返回MSE对w和b的梯度。"""
    n = X.shape[0]
    y_pred = predict(X, w, b)
    error = y_pred - y
    gradient_w = (2/n) * X.T @ error
    gradient_b = (2/n) * np.sum(error)
    return gradient_w, gradient_b


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
    if initial_w is None:
        w = np.zeros(X.shape[1], dtype = float)
    else:
        w = initial_w.astype(float).copy()

    b = initial_b

    loss_history = [mean_squared_error(y, predict(X, w, b))]

    for _ in range (n_steps):
        g_w, g_b = mse_gradients(X, y, w, b)

        w -= g_w * learning_rate
        b -= g_b * learning_rate

        tmp = mean_squared_error(y, predict(X, w, b))
        loss_history.append(tmp)

    return w, b, np.array(loss_history)
