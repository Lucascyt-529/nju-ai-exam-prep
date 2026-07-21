"""参考实现：线性核二分类SVM的确定性简化SMO。"""

import numpy as np


MODEL_KEYS = {
    "X_train",
    "y_train",
    "alphas",
    "bias",
    "C",
    "iterations",
    "updates",
    "dual_history",
}


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


def _validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
    _validate_matrix(X, "X")
    if (
        not _is_finite_numeric_array(y)
        or y.shape != (X.shape[0],)
        or set(np.unique(y).tolist()) != {-1, 1}
    ):
        raise ValueError("y必须是形状(n,)且同时包含-1和+1")


def _positive_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    valid = (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def linear_kernel_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    _validate_matrix(X, "X")
    _validate_matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X和Z特征数必须一致")
    return X.astype(float, copy=False) @ Z.astype(float, copy=False).T


def dual_objective(alphas: np.ndarray, y: np.ndarray, gram: np.ndarray) -> float:
    if (
        not _is_finite_numeric_array(alphas)
        or alphas.ndim != 1
        or np.any(alphas < 0)
        or not _is_finite_numeric_array(y)
        or y.shape != alphas.shape
        or not np.all(np.isin(y, [-1, 1]))
        or not _is_finite_numeric_array(gram)
        or gram.shape != (alphas.size, alphas.size)
        or alphas.size == 0
    ):
        raise ValueError("alphas、y和gram的形状或数值无效")
    weighted = alphas * y
    return float(np.sum(alphas) - 0.5 * weighted @ gram @ weighted)


def _pair_bounds(
    alpha_i: float, alpha_j: float, y_i: float, y_j: float, C: float
) -> tuple[float, float]:
    if y_i != y_j:
        return max(0.0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i)
    return max(0.0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j)


def _objective_with_pair(
    alphas: np.ndarray,
    i: int,
    j: int,
    alpha_i: float,
    alpha_j: float,
    y: np.ndarray,
    gram: np.ndarray,
) -> float:
    candidate = alphas.copy()
    candidate[i] = alpha_i
    candidate[j] = alpha_j
    return dual_objective(candidate, y, gram)


def _try_pair_update(
    i: int,
    j: int,
    alphas: np.ndarray,
    bias: float,
    y: np.ndarray,
    gram: np.ndarray,
    C: float,
    error_i: float,
    error_j: float,
    change_tolerance: float = 1e-10,
) -> tuple[bool, float]:
    if i == j:
        return False, bias
    old_i, old_j = float(alphas[i]), float(alphas[j])
    lower, upper = _pair_bounds(old_i, old_j, y[i], y[j], C)
    if upper - lower <= change_tolerance:
        return False, bias

    eta = 2.0 * gram[i, j] - gram[i, i] - gram[j, j]
    if eta < -1e-15:
        new_j = old_j - y[j] * (error_i - error_j) / eta
        new_j = float(np.clip(new_j, lower, upper))
    else:
        candidates = []
        for endpoint in (lower, upper):
            endpoint_i = old_i + y[i] * y[j] * (old_j - endpoint)
            value = _objective_with_pair(
                alphas, i, j, endpoint_i, endpoint, y, gram
            )
            candidates.append((value, endpoint))
        current = dual_objective(alphas, y, gram)
        best_value, new_j = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_value <= current + change_tolerance:
            return False, bias

    if abs(new_j - old_j) <= change_tolerance:
        return False, bias
    new_i = old_i + y[i] * y[j] * (old_j - new_j)
    if new_i < -1e-8 or new_i > C + 1e-8:
        return False, bias
    new_i = float(np.clip(new_i, 0.0, C))

    b1 = (
        bias
        - error_i
        - y[i] * (new_i - old_i) * gram[i, i]
        - y[j] * (new_j - old_j) * gram[i, j]
    )
    b2 = (
        bias
        - error_j
        - y[i] * (new_i - old_i) * gram[i, j]
        - y[j] * (new_j - old_j) * gram[j, j]
    )
    alphas[i], alphas[j] = new_i, new_j
    if change_tolerance < new_i < C - change_tolerance:
        new_bias = b1
    elif change_tolerance < new_j < C - change_tolerance:
        new_bias = b2
    else:
        new_bias = 0.5 * (b1 + b2)
    return True, float(new_bias)


def fit_linear_svm_smo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    tolerance: float = 1e-3,
    max_passes: int = 10,
    max_iterations: int = 1000,
) -> dict[str, object]:
    _validate_training_data(X, y)
    C_value = _positive_scalar(C, "C")
    tolerance_value = _positive_scalar(tolerance, "tolerance")
    max_passes_value = _positive_integer(max_passes, "max_passes")
    max_iterations_value = _positive_integer(max_iterations, "max_iterations")
    X_float = X.astype(float, copy=True)
    y_float = y.astype(float, copy=True)
    gram = linear_kernel_matrix(X_float, X_float)
    alphas = np.zeros(X.shape[0], dtype=float)
    bias = 0.0
    passes_without_change = 0
    iterations = 0
    total_updates = 0
    history = [dual_objective(alphas, y_float, gram)]

    while passes_without_change < max_passes_value and iterations < max_iterations_value:
        changed = 0
        for i in range(X.shape[0]):
            scores = gram @ (alphas * y_float) + bias
            errors = scores - y_float
            error_i = float(errors[i])
            violates = (
                y_float[i] * error_i < -tolerance_value
                and alphas[i] < C_value - 1e-10
            ) or (
                y_float[i] * error_i > tolerance_value
                and alphas[i] > 1e-10
            )
            if not violates:
                continue
            candidates = np.arange(X.shape[0])
            candidates = candidates[candidates != i]
            differences = np.abs(error_i - errors[candidates])
            order = np.lexsort((candidates, -differences))
            for j in candidates[order]:
                updated, new_bias = _try_pair_update(
                    i,
                    int(j),
                    alphas,
                    bias,
                    y_float,
                    gram,
                    C_value,
                    error_i,
                    float(errors[j]),
                )
                if updated:
                    bias = new_bias
                    changed += 1
                    total_updates += 1
                    break
        iterations += 1
        history.append(dual_objective(alphas, y_float, gram))
        if changed == 0:
            passes_without_change += 1
        else:
            passes_without_change = 0

    return {
        "X_train": X_float,
        "y_train": y_float,
        "alphas": alphas,
        "bias": bias,
        "C": C_value,
        "iterations": iterations,
        "updates": total_updates,
        "dual_history": np.array(history),
    }


def _validate_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("model键集合无效")
    X = model["X_train"]
    y = model["y_train"]
    alphas = model["alphas"]
    _validate_training_data(X, y)
    C = _positive_scalar(model["C"], "model C")
    if (
        not _is_finite_numeric_array(alphas)
        or alphas.shape != (X.shape[0],)
        or np.any(alphas < -1e-10)
        or np.any(alphas > C + 1e-10)
    ):
        raise ValueError("model alphas无效")
    if not isinstance(model["bias"], (int, float, np.integer, np.floating)) or not np.isfinite(model["bias"]):
        raise ValueError("model bias无效")


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model)
    _validate_matrix(X, "X")
    if X.shape[1] != model["X_train"].shape[1]:
        raise ValueError("X特征数与训练数据不一致")
    kernel = linear_kernel_matrix(X, model["X_train"])
    return kernel @ (model["alphas"] * model["y_train"]) + float(model["bias"])


def predict_labels(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.where(decision_function(model, X) >= 0.0, 1, -1)


def linear_weights(model: dict[str, object]) -> np.ndarray:
    _validate_model(model)
    return model["X_train"].T @ (model["alphas"] * model["y_train"])


def support_vector_indices(
    model: dict[str, object], *, alpha_tolerance: float = 1e-7
) -> np.ndarray:
    _validate_model(model)
    tolerance_value = _positive_scalar(alpha_tolerance, "alpha_tolerance")
    return np.flatnonzero(model["alphas"] > tolerance_value)


def kkt_residuals(
    model: dict[str, object], *, alpha_tolerance: float = 1e-7
) -> np.ndarray:
    _validate_model(model)
    tolerance_value = _positive_scalar(alpha_tolerance, "alpha_tolerance")
    margins = model["y_train"] * decision_function(model, model["X_train"])
    alphas = model["alphas"]
    residuals = np.empty_like(alphas)
    at_lower = alphas <= tolerance_value
    at_upper = alphas >= float(model["C"]) - tolerance_value
    free = ~(at_lower | at_upper)
    residuals[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
    residuals[at_upper] = np.maximum(0.0, margins[at_upper] - 1.0)
    residuals[free] = np.abs(margins[free] - 1.0)
    return residuals


def primal_objective(model: dict[str, object]) -> float:
    _validate_model(model)
    weights = linear_weights(model)
    margins = model["y_train"] * decision_function(model, model["X_train"])
    hinge = np.maximum(0.0, 1.0 - margins)
    return 0.5 * float(weights @ weights) + float(model["C"]) * float(np.sum(hinge))
