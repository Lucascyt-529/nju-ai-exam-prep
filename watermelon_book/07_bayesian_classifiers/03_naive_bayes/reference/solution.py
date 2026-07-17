"""参考实现：离散与高斯朴素贝叶斯。"""

import numpy as np


def _positive(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _validate_y(y: np.ndarray, n: int) -> None:
    if not isinstance(y, np.ndarray) or y.shape != (n,) or y.size == 0:
        raise ValueError("y必须是形状(n,)的一维标签数组")
    if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
        raise ValueError("y不能包含非有限值")
    if np.unique(y).size < 2:
        raise ValueError("y必须至少包含两个类别")


def _validate_categorical_X(X: np.ndarray, name: str = "X") -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError(f"{name}必须是非空二维数组")
    for value in X.flat:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError(f"{name}不能包含None或非有限值")


def _validate_numeric_X(X: np.ndarray, name: str = "X") -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def fit_categorical_nb(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> dict[str, object]:
    _validate_categorical_X(X)
    _validate_y(y, X.shape[0])
    alpha_value = _positive(alpha, "alpha")
    classes, counts = np.unique(y, return_counts=True)
    K, d = classes.size, X.shape[1]
    class_log_prior = np.log((counts + alpha_value) / (X.shape[0] + alpha_value * K))
    categories, log_probabilities, unknown_log_probabilities = [], [], []
    for feature in range(d):
        values = np.unique(X[:, feature])
        table = np.empty((K, values.size), dtype=float)
        unknown = np.empty(K, dtype=float)
        for class_index, label in enumerate(classes):
            class_values = X[y == label, feature]
            denominator = class_values.size + alpha_value * (values.size + 1)
            table[class_index] = np.log(
                (np.array([np.count_nonzero(class_values == value) for value in values]) + alpha_value) / denominator
            )
            unknown[class_index] = np.log(alpha_value / denominator)
        categories.append(values.copy())
        log_probabilities.append(table)
        unknown_log_probabilities.append(unknown)
    return {"kind": "categorical", "classes": classes.copy(), "class_log_prior": class_log_prior,
            "categories": tuple(categories), "feature_log_prob": tuple(log_probabilities),
            "unknown_log_prob": tuple(unknown_log_probabilities), "alpha": alpha_value, "n_features": d}


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray, *, variance_floor: float = 1e-9) -> dict[str, object]:
    _validate_numeric_X(X)
    _validate_y(y, X.shape[0])
    floor = _positive(variance_floor, "variance_floor")
    classes, counts = np.unique(y, return_counts=True)
    means = np.vstack([np.mean(X[y == label], axis=0) for label in classes])
    variances = np.vstack([np.var(X[y == label], axis=0, ddof=0) for label in classes])
    variances = np.maximum(variances, floor)
    return {"kind": "gaussian", "classes": classes.copy(), "class_log_prior": np.log(counts / X.shape[0]),
            "means": means, "variances": variances, "variance_floor": floor, "n_features": X.shape[1]}


def _validate_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or model.get("kind") not in {"categorical", "gaussian"}:
        raise ValueError("model不是有效的朴素贝叶斯模型")
    classes, priors = model.get("classes"), model.get("class_log_prior")
    if not isinstance(classes, np.ndarray) or classes.ndim != 1 or classes.size < 2 or not isinstance(priors, np.ndarray) or priors.shape != classes.shape or not np.all(np.isfinite(priors)):
        raise ValueError("model类别或先验无效")


def joint_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model)
    if model["kind"] == "categorical":
        _validate_categorical_X(X)
    else:
        _validate_numeric_X(X)
    if X.shape[1] != model["n_features"]:
        raise ValueError("X特征数与模型不一致")
    scores = np.tile(model["class_log_prior"], (X.shape[0], 1))
    if model["kind"] == "categorical":
        for feature in range(X.shape[1]):
            values = model["categories"][feature]
            table = model["feature_log_prob"][feature]
            unknown = model["unknown_log_prob"][feature]
            for row, value in enumerate(X[:, feature]):
                matches = np.flatnonzero(values == value)
                scores[row] += table[:, matches[0]] if matches.size else unknown
    else:
        means, variances = model["means"], model["variances"]
        differences = X.astype(float)[:, None, :] - means[None, :, :]
        scores += -0.5 * np.sum(np.log(2.0 * np.pi * variances)[None, :, :] + differences**2 / variances[None, :, :], axis=2)
    return scores


def predict_proba(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    scores = joint_log_scores(model, X)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    scores = joint_log_scores(model, X)
    return model["classes"][np.argmax(scores, axis=1)]
