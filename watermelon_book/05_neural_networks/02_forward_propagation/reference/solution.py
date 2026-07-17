"""参考实现：单隐层网络的初始化、前向传播和稳定交叉熵。"""

import numpy as np


PARAMETER_KEYS = {"W1", "b1", "W2", "b2"}


def _is_positive_integer(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value > 0
    )


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


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _validate_parameters(parameters: dict[str, np.ndarray], n_features: int) -> None:
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("parameters必须恰好包含W1、b1、W2和b2")
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )
    if not all(_is_finite_numeric_array(value) for value in (W1, b1, W2, b2)):
        raise ValueError("所有参数必须是有限数值NumPy数组")
    if W1.ndim != 2 or W1.shape[0] != n_features or W1.shape[1] == 0:
        raise ValueError("W1必须具有形状(d,h)，且h为正")
    n_hidden = W1.shape[1]
    if b1.shape != (n_hidden,) or W2.shape != (n_hidden, 1) or b2.shape != (1,):
        raise ValueError("参数形状必须是W1:(d,h)、b1:(h,)、W2:(h,1)、b2:(1,)")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or values.size == 0
        or not np.issubdtype(values.dtype, np.number)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("values必须是非空有限数值NumPy数组")
    values_float = values.astype(float, copy=False)
    result = np.empty_like(values_float)
    nonnegative = values_float >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values_float[nonnegative]))
    exponent = np.exp(values_float[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def initialize_parameters(
    n_features: int, n_hidden: int, *, seed: int = 0
) -> dict[str, np.ndarray]:
    if not _is_positive_integer(n_features) or not _is_positive_integer(n_hidden):
        raise ValueError("n_features和n_hidden必须是正整数")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed必须是整数")
    generator = np.random.default_rng(int(seed))
    W1 = generator.normal(0.0, 1.0 / np.sqrt(n_features), size=(n_features, n_hidden))
    W2 = generator.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1))
    return {
        "W1": W1,
        "b1": np.zeros(n_hidden),
        "W2": W2,
        "b2": np.zeros(1),
    }


def as_column_labels(y: np.ndarray) -> np.ndarray:
    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 1
        or y.size == 0
        or not np.issubdtype(y.dtype, np.number)
        or not np.all(np.isfinite(y))
        or not np.all(np.isin(y, [0, 1]))
    ):
        raise ValueError("y必须是非空的一维0/1数值数组")
    return y.astype(float, copy=True).reshape(-1, 1)


def forward_pass(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    _validate_X(X)
    _validate_parameters(parameters, X.shape[1])
    X_float = X.astype(float, copy=True)
    z1 = X_float @ parameters["W1"] + parameters["b1"]
    a1 = stable_sigmoid(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    probabilities = stable_sigmoid(z2)
    return {
        "X": X_float,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "probabilities": probabilities,
    }


def binary_cross_entropy_from_logits(y_column: np.ndarray, logits: np.ndarray) -> float:
    valid_pair = (
        _is_finite_numeric_array(y_column)
        and _is_finite_numeric_array(logits)
        and y_column.ndim == 2
        and y_column.shape[0] > 0
        and y_column.shape[1] == 1
        and logits.shape == y_column.shape
        and np.all(np.isin(y_column, [0, 1]))
    )
    if not valid_pair:
        raise ValueError("y_column和logits必须是同形状(n,1)有限数组，且标签只能为0/1")
    y_float = y_column.astype(float, copy=False)
    logits_float = logits.astype(float, copy=False)
    losses = (
        np.maximum(logits_float, 0.0)
        - logits_float * y_float
        + np.log1p(np.exp(-np.abs(logits_float)))
    )
    return float(np.mean(losses))
