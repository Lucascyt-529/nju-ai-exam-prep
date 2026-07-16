"""参考实现：NumPy线性回归的预测、损失、梯度与拟合。"""

import numpy as np


def _validate_X(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")


def _validate_y(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y 必须具有形状 (n_samples,)，不能是二维列数组")
    if not np.all(np.isfinite(y)):
        raise ValueError("y 必须只包含有限数值")


def _validate_w(X: np.ndarray, w: np.ndarray) -> None:
    if not isinstance(w, np.ndarray) or w.ndim != 1 or w.shape[0] != X.shape[1]:
        raise ValueError("w 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(w)):
        raise ValueError("w 必须只包含有限数值")


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """返回形状为 (n_samples,) 的线性预测。"""
    _validate_X(X)
    _validate_w(X, w)
    if not np.isfinite(b):
        raise ValueError("b 必须是有限标量")
    return X @ w + b


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0 or true.shape != pred.shape:
        raise ValueError("y_true 和 y_pred 必须是形状一致的非空一维数组")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("目标和预测必须只包含有限数值")
    return float(np.mean((pred - true) ** 2))


def mse_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """返回MSE对w和b的梯度。"""
    _validate_X(X)
    _validate_y(X, y)
    _validate_w(X, w)
    predictions = predict(X, w, b)
    errors = predictions - y
    gradient_w = (2.0 / X.shape[0]) * (X.T @ errors)
    gradient_b = float(2.0 * errors.mean())
    return gradient_w, gradient_b


def fit_least_squares(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """使用 np.linalg.lstsq 拟合线性回归。"""
    _validate_X(X)
    _validate_y(X, y)
    if fit_intercept:
        design = np.column_stack((np.ones(X.shape[0], dtype=float), X))
        parameters, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return parameters[1:].copy(), float(parameters[0])
    weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return weights.copy(), 0.0


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
    _validate_X(X)
    _validate_y(X, y)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须是正有限数值")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps 必须是正整数")
    if not np.isfinite(initial_b):
        raise ValueError("initial_b 必须是有限标量")

    if initial_w is None:
        w = np.zeros(X.shape[1], dtype=float)
    else:
        _validate_w(X, initial_w)
        w = initial_w.astype(float, copy=True)
    b = float(initial_b)
    losses = np.empty(n_steps + 1, dtype=float)
    losses[0] = mean_squared_error(y, predict(X, w, b))

    for step in range(1, n_steps + 1):
        gradient_w, gradient_b = mse_gradients(X, y, w, b)
        w -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
        loss = mean_squared_error(y, predict(X, w, b))
        if not np.isfinite(loss):
            raise FloatingPointError("梯度下降发散：损失变为非有限数值")
        losses[step] = loss
    return w, b, losses
