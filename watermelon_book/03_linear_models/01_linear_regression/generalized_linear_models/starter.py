"""学生练习：联系函数、逆联系与对数线性回归。"""

import numpy as np


def linear_predictor(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    raise NotImplementedError("请完成 linear_predictor")


def apply_link(values: np.ndarray, *, link: str) -> np.ndarray:
    raise NotImplementedError("请完成 apply_link")


def inverse_link(eta: np.ndarray, *, link: str) -> np.ndarray:
    raise NotImplementedError("请完成 inverse_link")


def predict_mean(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    link: str,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_mean")


def fit_log_linear(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 fit_log_linear")


def predict_log_linear(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_log_linear")


def mean_squared_link_error(
    y_true: np.ndarray, y_pred: np.ndarray, *, link: str
) -> float:
    raise NotImplementedError("请完成 mean_squared_link_error")
