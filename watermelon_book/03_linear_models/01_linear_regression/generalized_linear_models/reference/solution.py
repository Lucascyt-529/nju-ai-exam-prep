"""参考实现：联系函数、逆联系与对数线性回归小实验。"""

import numpy as np


VALID_LINKS = {"identity", "log", "logit"}


def _finite_numeric_array(values: object, name: str) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
        or values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name}必须是非空有限数值NumPy数组")
    return values.astype(float, copy=False)


def _validate_X(X: np.ndarray) -> np.ndarray:
    values = _finite_numeric_array(X, "X")
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("X必须是非空有限数值二维数组")
    return values


def _validate_y(X: np.ndarray, y: np.ndarray, *, positive: bool = False) -> np.ndarray:
    values = _finite_numeric_array(y, "y")
    if values.shape != (X.shape[0],):
        raise ValueError("y必须具有形状(n_samples,)")
    if positive and np.any(values <= 0):
        raise ValueError("对数联系要求y全部严格大于0")
    return values


def _validate_parameters(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    values = _finite_numeric_array(weights, "weights")
    if values.shape != (X.shape[1],):
        raise ValueError("weights必须具有形状(n_features,)")
    if (
        not isinstance(bias, (int, float, np.integer, np.floating))
        or isinstance(bias, (bool, np.bool_))
        or not np.isfinite(bias)
    ):
        raise ValueError("bias必须是有限数值标量")
    return values, float(bias)


def linear_predictor(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """计算eta=X@weights+bias，返回形状(n_samples,)。"""
    X_float = _validate_X(X)
    weights_float, bias_float = _validate_parameters(X_float, weights, bias)
    return X_float @ weights_float + bias_float


def apply_link(values: np.ndarray, *, link: str) -> np.ndarray:
    """计算g(mu)：恒等、对数或logit联系。"""
    array = _finite_numeric_array(values, "values")
    if link not in VALID_LINKS:
        raise ValueError(f"link必须是{sorted(VALID_LINKS)}之一")
    if link == "identity":
        return array.astype(float, copy=True)
    if link == "log":
        if np.any(array <= 0):
            raise ValueError("log联系要求values全部严格大于0")
        return np.log(array)
    if np.any((array <= 0) | (array >= 1)):
        raise ValueError("logit联系要求values全部严格位于(0,1)")
    return np.log(array) - np.log1p(-array)


def inverse_link(eta: np.ndarray, *, link: str) -> np.ndarray:
    """计算g^{-1}(eta)，并拒绝指数逆联系溢出。"""
    values = _finite_numeric_array(eta, "eta")
    if link not in VALID_LINKS:
        raise ValueError(f"link必须是{sorted(VALID_LINKS)}之一")
    if link == "identity":
        return values.astype(float, copy=True)
    if link == "log":
        with np.errstate(over="ignore"):
            result = np.exp(values)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("log逆联系exp(eta)发生溢出")
        return result

    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def predict_mean(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    link: str,
) -> np.ndarray:
    """先计算线性预测子eta，再通过逆联系得到条件均值mu。"""
    eta = linear_predictor(X, weights, bias)
    return inverse_link(eta, link=link)


def fit_log_linear(
    X: np.ndarray, y: np.ndarray, *, fit_intercept: bool = True
) -> tuple[np.ndarray, float]:
    """对log(y)做最小二乘，得到log(mu)=Xw+b的透明教学实现。"""
    X_float = _validate_X(X)
    y_float = _validate_y(X_float, y, positive=True)
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept必须是布尔值")
    target = apply_link(y_float, link="log")
    if bool(fit_intercept):
        design = np.column_stack((np.ones(X_float.shape[0]), X_float))
        parameters, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return parameters[1:].copy(), float(parameters[0])
    weights, _, _, _ = np.linalg.lstsq(X_float, target, rcond=None)
    return weights.copy(), 0.0


def predict_log_linear(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    return predict_mean(X, weights, bias, link="log")


def mean_squared_link_error(
    y_true: np.ndarray, y_pred: np.ndarray, *, link: str
) -> float:
    """在指定联系函数变换后的尺度上计算均方误差。"""
    true = _finite_numeric_array(y_true, "y_true")
    predicted = _finite_numeric_array(y_pred, "y_pred")
    if true.ndim != 1 or predicted.shape != true.shape:
        raise ValueError("y_true和y_pred必须是同形状的非空一维数组")
    linked_true = apply_link(true, link=link)
    linked_predicted = apply_link(predicted, link=link)
    return float(np.mean((linked_predicted - linked_true) ** 2))
