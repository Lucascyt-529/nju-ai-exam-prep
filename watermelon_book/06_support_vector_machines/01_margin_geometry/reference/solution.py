"""参考实现：线性超平面的函数间隔与几何间隔。"""

import numpy as np


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _validate_X_weights_bias(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> None:
    valid_X = (
        _is_finite_numeric_array(X)
        and X.ndim == 2
        and X.shape[0] > 0
        and X.shape[1] > 0
    )
    valid_weights = (
        valid_X
        and _is_finite_numeric_array(weights)
        and weights.shape == (X.shape[1],)
    )
    valid_bias = (
        isinstance(bias, (int, float, np.integer, np.floating))
        and not isinstance(bias, (bool, np.bool_))
        and np.isfinite(bias)
    )
    if not valid_X or not valid_weights or not valid_bias:
        raise ValueError("X必须是(n,d)，weights必须是(d,)，bias必须是有限标量")


def _validate_y(y: np.ndarray, n_samples: int) -> None:
    if (
        not _is_finite_numeric_array(y)
        or y.shape != (n_samples,)
        or set(np.unique(y).tolist()) != {-1, 1}
    ):
        raise ValueError("y必须是形状(n,)且同时包含-1和+1")


def _weight_norm(weights: np.ndarray) -> float:
    norm = float(np.linalg.norm(weights))
    if norm == 0.0:
        raise ValueError("weights不能是零向量")
    return norm


def decision_scores(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    _validate_X_weights_bias(X, weights, bias)
    return X.astype(float, copy=False) @ weights.astype(float, copy=False) + float(bias)


def predict_labels(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return np.where(decision_scores(X, weights, bias) >= 0.0, 1, -1)


def functional_margins(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    scores = decision_scores(X, weights, bias)
    _validate_y(y, X.shape[0])
    return y.astype(float, copy=False) * scores


def geometric_margins(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    margins = functional_margins(X, y, weights, bias)
    return margins / _weight_norm(weights)


def point_to_hyperplane_distances(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    scores = decision_scores(X, weights, bias)
    return np.abs(scores) / _weight_norm(weights)


def canonical_rescale(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    margins = functional_margins(X, y, weights, bias)
    minimum = float(np.min(margins))
    if minimum <= 0.0:
        raise ValueError("只有正确分开全部样本的超平面才能规范化")
    return weights.astype(float, copy=True) / minimum, float(bias) / minimum


def minimum_margin_indices(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    margins = geometric_margins(X, y, weights, bias)
    valid_tolerance = (
        isinstance(tolerance, (int, float, np.integer, np.floating))
        and not isinstance(tolerance, (bool, np.bool_))
        and np.isfinite(tolerance)
        and tolerance >= 0
    )
    if not valid_tolerance:
        raise ValueError("tolerance必须是非负有限数值")
    minimum = float(np.min(margins))
    return np.flatnonzero(np.abs(margins - minimum) <= float(tolerance))


def hard_margin_primal_objective(weights: np.ndarray) -> float:
    if (
        not _is_finite_numeric_array(weights)
        or weights.ndim != 1
        or weights.size == 0
    ):
        raise ValueError("weights必须是非空一维有限数值数组")
    return 0.5 * float(weights @ weights)
