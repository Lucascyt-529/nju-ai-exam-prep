"""参考实现：确定性二分类感知机。"""

import numpy as np


def _validate_X(X: np.ndarray) -> None:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not np.issubdtype(X.dtype, np.number)
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X必须是非空有限数值二维数组")


def _validate_xy(X: np.ndarray, y: np.ndarray) -> None:
    _validate_X(X)
    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 1
        or y.shape[0] != X.shape[0]
        or not np.issubdtype(y.dtype, np.number)
        or not np.all(np.isfinite(y))
        or set(np.unique(y).tolist()) != {-1, 1}
    ):
        raise ValueError("y必须是一维、样本数匹配并且同时包含-1和+1")


def _validate_parameters(weights: np.ndarray, bias: float, n_features: int) -> None:
    valid_weights = (
        isinstance(weights, np.ndarray)
        and weights.shape == (n_features,)
        and np.issubdtype(weights.dtype, np.number)
        and np.all(np.isfinite(weights))
    )
    valid_bias = (
        isinstance(bias, (int, float, np.integer, np.floating))
        and not isinstance(bias, (bool, np.bool_))
        and np.isfinite(bias)
    )
    if not valid_weights or not valid_bias:
        raise ValueError("weights必须是形状(d,)的有限数值数组，bias必须是有限标量")


def decision_function(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    _validate_X(X)
    _validate_parameters(weights, bias, X.shape[1])
    return X.astype(float, copy=False) @ weights.astype(float, copy=False) + float(bias)


def predict_perceptron(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    scores = decision_function(X, weights, bias)
    return np.where(scores >= 0.0, 1, -1)


def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 1.0,
    max_epochs: int = 100,
) -> tuple[np.ndarray, float, list[int]]:
    _validate_xy(X, y)
    valid_learning_rate = (
        isinstance(learning_rate, (int, float, np.integer, np.floating))
        and not isinstance(learning_rate, (bool, np.bool_))
        and np.isfinite(learning_rate)
        and learning_rate > 0
    )
    valid_max_epochs = (
        isinstance(max_epochs, (int, np.integer))
        and not isinstance(max_epochs, (bool, np.bool_))
        and max_epochs > 0
    )
    if not valid_learning_rate or not valid_max_epochs:
        raise ValueError("learning_rate必须是正有限数值，max_epochs必须是正整数")

    X_float = X.astype(float, copy=False)
    y_integer = y.astype(int, copy=False)
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    updates_history: list[int] = []

    for _ in range(int(max_epochs)):
        updates = 0
        for sample, label in zip(X_float, y_integer):
            score = float(sample @ weights + bias)
            prediction = 1 if score >= 0.0 else -1
            if prediction != label:
                correction = float(learning_rate) * int(label)
                weights += correction * sample
                bias += correction
                updates += 1
        updates_history.append(updates)
        if updates == 0:
            break
    return weights, bias, updates_history
