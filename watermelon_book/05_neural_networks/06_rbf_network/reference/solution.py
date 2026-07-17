"""参考实现：高斯RBF隐藏层与线性输出拟合。"""

import numpy as np


MODEL_KEYS = {"centers", "width", "output_weights"}


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _validate_matrix(value: np.ndarray, name: str) -> None:
    if (
        not _is_finite_numeric_array(value)
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] == 0
    ):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _positive_finite_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _nonnegative_finite_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value >= 0
    )
    if not valid:
        raise ValueError(f"{name}必须是非负有限数值")
    return float(value)


def squared_euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    _validate_matrix(X, "X")
    _validate_matrix(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X和centers的特征数必须一致")
    difference = X.astype(float, copy=False)[:, None, :] - centers.astype(float, copy=False)[None, :, :]
    return np.sum(difference**2, axis=2)


def rbf_design_matrix(
    X: np.ndarray,
    centers: np.ndarray,
    width: float,
    *,
    include_bias: bool = True,
) -> np.ndarray:
    width_value = _positive_finite_scalar(width, "width")
    if not isinstance(include_bias, (bool, np.bool_)):
        raise ValueError("include_bias必须是布尔值")
    distances = squared_euclidean_distances(X, centers)
    responses = np.exp(-distances / (2.0 * width_value**2))
    if bool(include_bias):
        return np.column_stack([responses, np.ones(X.shape[0])])
    return responses


def fit_rbf_output(
    X: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    width: float,
    *,
    regularization: float = 0.0,
) -> dict[str, np.ndarray | float]:
    _validate_matrix(X, "X")
    _validate_matrix(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X和centers的特征数必须一致")
    if (
        not _is_finite_numeric_array(y)
        or y.ndim != 1
        or y.shape[0] != X.shape[0]
    ):
        raise ValueError("y必须是与X样本数一致的一维有限数值数组")
    width_value = _positive_finite_scalar(width, "width")
    regularization_value = _nonnegative_finite_scalar(regularization, "regularization")
    design = rbf_design_matrix(X, centers, width_value)
    y_float = y.astype(float, copy=False)
    if regularization_value == 0.0:
        output_weights = np.linalg.lstsq(design, y_float, rcond=None)[0]
    else:
        penalty = np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        output_weights = np.linalg.solve(
            design.T @ design + regularization_value * penalty,
            design.T @ y_float,
        )
    return {
        "centers": centers.astype(float, copy=True),
        "width": width_value,
        "output_weights": output_weights,
    }


def _validate_model(model: dict[str, np.ndarray | float], n_features: int) -> None:
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("model必须恰好包含centers、width和output_weights")
    centers = model["centers"]
    _validate_matrix(centers, "model centers")
    if centers.shape[1] != n_features:
        raise ValueError("模型中心特征数与X不一致")
    _positive_finite_scalar(model["width"], "model width")
    weights = model["output_weights"]
    if (
        not _is_finite_numeric_array(weights)
        or weights.shape != (centers.shape[0] + 1,)
    ):
        raise ValueError("output_weights必须具有形状(m+1,)")


def predict_rbf(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    _validate_matrix(X, "X")
    _validate_model(model, X.shape[1])
    design = rbf_design_matrix(X, model["centers"], model["width"])
    return design @ model["output_weights"]


def mean_squared_error(y_true: np.ndarray, prediction: np.ndarray) -> float:
    if (
        not _is_finite_numeric_array(y_true)
        or not _is_finite_numeric_array(prediction)
        or y_true.ndim != 1
        or prediction.shape != y_true.shape
        or y_true.size == 0
    ):
        raise ValueError("y_true和prediction必须是同形状非空一维有限数组")
    return float(np.mean((prediction - y_true) ** 2))
