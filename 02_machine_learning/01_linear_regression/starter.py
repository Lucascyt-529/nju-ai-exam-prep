"""学生练习：按 predict -> MSE -> gradients -> GD -> least squares 实现。"""

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """计算线性预测。

    输入：特征 X、权重 w 和截距 b。
    输出：每个样本的预测值。
    """
    raise NotImplementedError("请完成 predict")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均平方误差。

    输入：真实值 y_true 和预测值 y_pred。
    输出：一个 Python float。
    """
    raise NotImplementedError("请完成 mean_squared_error")


def mse_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """计算 MSE 对 w 和 b 的解析梯度。

    输入：训练数据 X、y 和当前参数 w、b。
    输出：权重梯度 gradient_w 和截距梯度 gradient_b。
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

    输入：训练数据、学习率、迭代次数和可选初始参数。
    输出：拟合后的 w、b 和训练过程中的 loss_history。
    """
    raise NotImplementedError("请完成 fit_gradient_descent")


def fit_least_squares(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """使用 np.linalg.lstsq 拟合线性回归。

    输入：训练数据 X、y；fit_intercept 决定是否拟合截距。
    输出：拟合后的 w 和 b。
    """
    raise NotImplementedError("请完成 fit_least_squares")
