"""参考实现：单隐层网络BP与中心有限差分梯度检查。"""

import numpy as np


PARAMETER_KEYS = {"W1", "b1", "W2", "b2"}
CACHE_KEYS = {"X", "z1", "a1", "z2", "probabilities"}


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _validate_X_y(X: np.ndarray, y_column: np.ndarray) -> None:
    valid_X = (
        _is_finite_numeric_array(X)
        and X.ndim == 2
        and X.shape[0] > 0
        and X.shape[1] > 0
    )
    if not valid_X:
        raise ValueError("X必须是非空有限数值二维数组")
    _validate_y_column(y_column, n_samples=X.shape[0])


def _validate_y_column(y_column: np.ndarray, *, n_samples: int | None = None) -> None:
    valid_y = (
        _is_finite_numeric_array(y_column)
        and y_column.ndim == 2
        and y_column.shape[0] > 0
        and y_column.shape[1] == 1
        and (n_samples is None or y_column.shape[0] == n_samples)
        and np.all(np.isin(y_column, [0, 1]))
    )
    if not valid_y:
        raise ValueError("y_column必须是非空(n,1)有限二进制标签数组")


def _validate_parameters(parameters: dict[str, np.ndarray], n_features: int) -> None:
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("parameters必须恰好包含W1、b1、W2和b2")
    if not all(_is_finite_numeric_array(parameters[key]) for key in PARAMETER_KEYS):
        raise ValueError("所有参数必须是有限数值NumPy数组")
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )
    if W1.ndim != 2 or W1.shape[0] != n_features or W1.shape[1] == 0:
        raise ValueError("W1必须具有形状(d,h)，且h为正")
    n_hidden = W1.shape[1]
    if b1.shape != (n_hidden,) or W2.shape != (n_hidden, 1) or b2.shape != (1,):
        raise ValueError("参数形状必须是W1:(d,h)、b1:(h,)、W2:(h,1)、b2:(1,)")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    if not _is_finite_numeric_array(values) or values.size == 0:
        raise ValueError("values必须是非空有限数值NumPy数组")
    values_float = values.astype(float, copy=False)
    result = np.empty_like(values_float)
    nonnegative = values_float >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values_float[nonnegative]))
    exponent = np.exp(values_float[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def forward_pass(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not _is_finite_numeric_array(X) or X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X必须是非空有限数值二维数组")
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
    _validate_y_column(y_column)
    if not _is_finite_numeric_array(logits) or logits.shape != y_column.shape:
        raise ValueError("logits必须与y_column具有相同的(n,1)形状")
    y_float = y_column.astype(float, copy=False)
    logits_float = logits.astype(float, copy=False)
    losses = (
        np.maximum(logits_float, 0.0)
        - logits_float * y_float
        + np.log1p(np.exp(-np.abs(logits_float)))
    )
    return float(np.mean(losses))


def loss_for_parameters(
    X: np.ndarray, y_column: np.ndarray, parameters: dict[str, np.ndarray]
) -> float:
    _validate_X_y(X, y_column)
    cache = forward_pass(X, parameters)
    return binary_cross_entropy_from_logits(y_column, cache["z2"])


def _validate_cache(
    parameters: dict[str, np.ndarray], cache: dict[str, np.ndarray], y_column: np.ndarray
) -> None:
    if not isinstance(cache, dict) or set(cache) != CACHE_KEYS:
        raise ValueError("cache必须恰好包含X、z1、a1、z2和probabilities")
    if not all(_is_finite_numeric_array(cache[key]) for key in CACHE_KEYS):
        raise ValueError("cache中的所有值必须是有限数值NumPy数组")
    _validate_X_y(cache["X"], y_column)
    _validate_parameters(parameters, cache["X"].shape[1])
    expected = forward_pass(cache["X"], parameters)
    if any(
        cache[key].shape != expected[key].shape
        or not np.allclose(cache[key], expected[key], rtol=1e-12, atol=1e-12)
        for key in CACHE_KEYS
    ):
        raise ValueError("cache的形状或数值与X和parameters不一致")


def backward_pass(
    parameters: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    y_column: np.ndarray,
) -> dict[str, np.ndarray]:
    _validate_cache(parameters, cache, y_column)
    sample_count = cache["X"].shape[0]
    dz2 = (cache["probabilities"] - y_column) / sample_count
    dW2 = cache["a1"].T @ dz2
    db2 = np.sum(dz2, axis=0)
    da1 = dz2 @ parameters["W2"].T
    dz1 = da1 * cache["a1"] * (1.0 - cache["a1"])
    dW1 = cache["X"].T @ dz1
    db1 = np.sum(dz1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def finite_difference_gradients(
    X: np.ndarray,
    y_column: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    epsilon: float = 1e-5,
) -> dict[str, np.ndarray]:
    _validate_X_y(X, y_column)
    _validate_parameters(parameters, X.shape[1])
    valid_epsilon = (
        isinstance(epsilon, (int, float, np.integer, np.floating))
        and not isinstance(epsilon, (bool, np.bool_))
        and np.isfinite(epsilon)
        and epsilon > 0
    )
    if not valid_epsilon:
        raise ValueError("epsilon必须是正有限数值")
    working = {key: value.astype(float, copy=True) for key, value in parameters.items()}
    gradients = {key: np.zeros_like(value) for key, value in working.items()}
    for key in ("W1", "b1", "W2", "b2"):
        for index in np.ndindex(working[key].shape):
            original = working[key][index]
            working[key][index] = original + float(epsilon)
            loss_plus = loss_for_parameters(X, y_column, working)
            working[key][index] = original - float(epsilon)
            loss_minus = loss_for_parameters(X, y_column, working)
            working[key][index] = original
            gradients[key][index] = (loss_plus - loss_minus) / (2.0 * float(epsilon))
    return gradients


def maximum_relative_error(
    analytic: dict[str, np.ndarray], numerical: dict[str, np.ndarray]
) -> float:
    if (
        not isinstance(analytic, dict)
        or not isinstance(numerical, dict)
        or set(analytic) != PARAMETER_KEYS
        or set(numerical) != PARAMETER_KEYS
    ):
        raise ValueError("两个梯度字典必须恰好包含W1、b1、W2和b2")
    maximum = 0.0
    for key in PARAMETER_KEYS:
        if (
            not _is_finite_numeric_array(analytic[key])
            or not _is_finite_numeric_array(numerical[key])
            or analytic[key].shape != numerical[key].shape
        ):
            raise ValueError("对应梯度必须是同形状有限数值数组")
        denominator = np.maximum(1e-12, np.abs(analytic[key]) + np.abs(numerical[key]))
        error = np.abs(analytic[key] - numerical[key]) / denominator
        maximum = max(maximum, float(np.max(error)))
    return maximum
