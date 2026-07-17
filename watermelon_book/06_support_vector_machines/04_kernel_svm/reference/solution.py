"""参考实现：线性、多项式、RBF核与确定性简化SMO。"""

import numpy as np


MODEL_KEYS = {
    "X_train", "y_train", "alphas", "bias", "C", "kernel", "degree",
    "gamma", "coef0", "iterations", "updates", "dual_history",
}
KERNELS = {"linear", "polynomial", "rbf"}


def _finite_array(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value))


def _matrix(value: np.ndarray, name: str) -> None:
    if not _finite_array(value) or value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _positive(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def _kernel_config(d: int, kernel: object, degree: object, gamma: object, coef0: object) -> tuple[str, int, float, float]:
    if kernel not in KERNELS:
        raise ValueError("kernel必须是linear、polynomial或rbf")
    degree_value = _positive_int(degree, "degree")
    gamma_value = 1.0 / d if gamma is None else _positive(gamma, "gamma")
    if not isinstance(coef0, (int, float, np.integer, np.floating)) or isinstance(coef0, (bool, np.bool_)) or not np.isfinite(coef0):
        raise ValueError("coef0必须是有限数值")
    return str(kernel), degree_value, gamma_value, float(coef0)


def kernel_matrix(
    X: np.ndarray, Z: np.ndarray, *, kernel: str = "linear", degree: int = 3,
    gamma: float | None = None, coef0: float = 1.0,
) -> np.ndarray:
    _matrix(X, "X")
    _matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X和Z特征数必须一致")
    kind, degree_value, gamma_value, coef0_value = _kernel_config(X.shape[1], kernel, degree, gamma, coef0)
    Xf, Zf = X.astype(float, copy=False), Z.astype(float, copy=False)
    dot = Xf @ Zf.T
    if kind == "linear":
        result = dot
    elif kind == "polynomial":
        with np.errstate(over="ignore", invalid="ignore"):
            result = (gamma_value * dot + coef0_value) ** degree_value
    else:
        squared = np.sum(Xf * Xf, axis=1)[:, None] + np.sum(Zf * Zf, axis=1)[None, :] - 2.0 * dot
        result = np.exp(-gamma_value * np.maximum(squared, 0.0))
    if not np.all(np.isfinite(result)):
        raise ValueError("核矩阵出现非有限数值，请调整数据或核参数")
    return result


def dual_objective(alphas: np.ndarray, y: np.ndarray, gram: np.ndarray) -> float:
    if not _finite_array(alphas) or alphas.ndim != 1 or alphas.size == 0 or np.any(alphas < 0) or not _finite_array(y) or y.shape != alphas.shape or not np.all(np.isin(y, [-1, 1])) or not _finite_array(gram) or gram.shape != (alphas.size, alphas.size):
        raise ValueError("alphas、y和gram无效")
    weighted = alphas * y
    return float(np.sum(alphas) - 0.5 * weighted @ gram @ weighted)


def _bounds(ai: float, aj: float, yi: float, yj: float, C: float) -> tuple[float, float]:
    if yi != yj:
        return max(0.0, aj - ai), min(C, C + aj - ai)
    return max(0.0, ai + aj - C), min(C, ai + aj)


def _try_update(i: int, j: int, a: np.ndarray, b: float, y: np.ndarray, K: np.ndarray, C: float, Ei: float, Ej: float) -> tuple[bool, float]:
    if i == j:
        return False, b
    old_i, old_j = float(a[i]), float(a[j])
    low, high = _bounds(old_i, old_j, y[i], y[j], C)
    if high - low <= 1e-10:
        return False, b
    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
    if eta < -1e-15:
        new_j = float(np.clip(old_j - y[j] * (Ei - Ej) / eta, low, high))
    else:
        current = dual_objective(a, y, K)
        choices = []
        for endpoint in (low, high):
            candidate = a.copy()
            candidate[j] = endpoint
            candidate[i] = old_i + y[i] * y[j] * (old_j - endpoint)
            choices.append((dual_objective(candidate, y, K), endpoint))
        best, new_j = max(choices, key=lambda item: (item[0], -item[1]))
        if best <= current + 1e-10:
            return False, b
    if abs(new_j - old_j) <= 1e-10:
        return False, b
    new_i = float(np.clip(old_i + y[i] * y[j] * (old_j - new_j), 0.0, C))
    b1 = b - Ei - y[i] * (new_i - old_i) * K[i, i] - y[j] * (new_j - old_j) * K[i, j]
    b2 = b - Ej - y[i] * (new_i - old_i) * K[i, j] - y[j] * (new_j - old_j) * K[j, j]
    a[i], a[j] = new_i, new_j
    if 1e-10 < new_i < C - 1e-10:
        return True, float(b1)
    if 1e-10 < new_j < C - 1e-10:
        return True, float(b2)
    return True, float(0.5 * (b1 + b2))


def fit_kernel_svm_smo(
    X: np.ndarray, y: np.ndarray, *, C: float = 1.0, kernel: str = "rbf",
    degree: int = 3, gamma: float | None = None, coef0: float = 1.0,
    tolerance: float = 1e-3, max_passes: int = 10, max_iterations: int = 1000,
) -> dict[str, object]:
    _matrix(X, "X")
    if not _finite_array(y) or y.shape != (X.shape[0],) or set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError("y必须是形状(n,)且同时包含-1和+1")
    C_value, tol = _positive(C, "C"), _positive(tolerance, "tolerance")
    passes_limit, iteration_limit = _positive_int(max_passes, "max_passes"), _positive_int(max_iterations, "max_iterations")
    kind, degree_value, gamma_value, coef0_value = _kernel_config(X.shape[1], kernel, degree, gamma, coef0)
    Xf, yf = X.astype(float, copy=True), y.astype(float, copy=True)
    K = kernel_matrix(Xf, Xf, kernel=kind, degree=degree_value, gamma=gamma_value, coef0=coef0_value)
    a, b, passes, iterations, updates = np.zeros(X.shape[0]), 0.0, 0, 0, 0
    history = [dual_objective(a, yf, K)]
    while passes < passes_limit and iterations < iteration_limit:
        changed = 0
        for i in range(X.shape[0]):
            errors = K @ (a * yf) + b - yf
            Ei = float(errors[i])
            violates = (yf[i] * Ei < -tol and a[i] < C_value - 1e-10) or (yf[i] * Ei > tol and a[i] > 1e-10)
            if not violates:
                continue
            candidates = np.arange(X.shape[0])
            candidates = candidates[candidates != i]
            order = np.lexsort((candidates, -np.abs(Ei - errors[candidates])))
            for j in candidates[order]:
                done, new_b = _try_update(i, int(j), a, b, yf, K, C_value, Ei, float(errors[j]))
                if done:
                    b, changed, updates = new_b, changed + 1, updates + 1
                    break
        iterations += 1
        history.append(dual_objective(a, yf, K))
        passes = passes + 1 if changed == 0 else 0
    return {"X_train": Xf, "y_train": yf, "alphas": a, "bias": b, "C": C_value,
            "kernel": kind, "degree": degree_value, "gamma": gamma_value, "coef0": coef0_value,
            "iterations": iterations, "updates": updates, "dual_history": np.array(history)}


def _validate_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("model键集合无效")
    _matrix(model["X_train"], "model X_train")
    n = model["X_train"].shape[0]
    if not _finite_array(model["y_train"]) or model["y_train"].shape != (n,) or not _finite_array(model["alphas"]) or model["alphas"].shape != (n,):
        raise ValueError("model训练向量无效")
    C = _positive(model["C"], "model C")
    if np.any(model["alphas"] < -1e-10) or np.any(model["alphas"] > C + 1e-10) or not np.isfinite(model["bias"]):
        raise ValueError("model参数无效")
    _kernel_config(model["X_train"].shape[1], model["kernel"], model["degree"], model["gamma"], model["coef0"])


def decision_function(model: dict[str, object], X: np.ndarray, *, support_only: bool = True) -> np.ndarray:
    _validate_model(model)
    _matrix(X, "X")
    if X.shape[1] != model["X_train"].shape[1]:
        raise ValueError("X特征数与训练数据不一致")
    indices = np.flatnonzero(model["alphas"] > 1e-10) if support_only else np.arange(model["alphas"].size)
    K = kernel_matrix(X, model["X_train"][indices], kernel=model["kernel"], degree=model["degree"], gamma=model["gamma"], coef0=model["coef0"])
    return K @ (model["alphas"][indices] * model["y_train"][indices]) + float(model["bias"])


def predict_labels(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.where(decision_function(model, X) >= 0.0, 1, -1)


def support_vector_indices(model: dict[str, object]) -> np.ndarray:
    _validate_model(model)
    return np.flatnonzero(model["alphas"] > 1e-10)


def kkt_residuals(model: dict[str, object], *, alpha_tolerance: float = 1e-7) -> np.ndarray:
    _validate_model(model)
    tol = _positive(alpha_tolerance, "alpha_tolerance")
    margins = model["y_train"] * decision_function(model, model["X_train"])
    a, C = model["alphas"], float(model["C"])
    lower, upper = a <= tol, a >= C - tol
    free = ~(lower | upper)
    result = np.empty_like(a)
    result[lower] = np.maximum(0.0, 1.0 - margins[lower])
    result[upper] = np.maximum(0.0, margins[upper] - 1.0)
    result[free] = np.abs(margins[free] - 1.0)
    return result
