"""参考实现：epsilon-SVR的确定性成对坐标优化。"""

import numpy as np


MODEL_KEYS = {"X_train", "y_train", "beta", "bias", "C", "epsilon", "kernel", "gamma", "passes", "updates", "dual_history"}


def _finite_array(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value))


def _vector(value: np.ndarray, name: str) -> None:
    if not _finite_array(value) or value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name}必须是非空有限数值一维数组")


def _matrix(value: np.ndarray, name: str) -> None:
    if not _finite_array(value) or value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _positive(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name}必须是非负有限数值")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def kernel_matrix(X: np.ndarray, Z: np.ndarray, *, kernel: str = "linear", gamma: float | None = None) -> np.ndarray:
    _matrix(X, "X")
    _matrix(Z, "Z")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X和Z特征数必须一致")
    if kernel not in {"linear", "rbf"}:
        raise ValueError("kernel必须是linear或rbf")
    gamma_value = 1.0 / X.shape[1] if gamma is None else _positive(gamma, "gamma")
    Xf, Zf = X.astype(float, copy=False), Z.astype(float, copy=False)
    dot = Xf @ Zf.T
    if kernel == "linear":
        return dot
    squared = np.sum(Xf * Xf, axis=1)[:, None] + np.sum(Zf * Zf, axis=1)[None, :] - 2.0 * dot
    return np.exp(-gamma_value * np.maximum(squared, 0.0))


def epsilon_insensitive_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> np.ndarray:
    _vector(y_true, "y_true")
    _vector(y_pred, "y_pred")
    epsilon_value = _nonnegative(epsilon, "epsilon")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true和y_pred形状必须一致")
    return np.maximum(0.0, np.abs(y_true.astype(float) - y_pred.astype(float)) - epsilon_value)


def tube_regions(residuals: np.ndarray, epsilon: float, *, tolerance: float = 1e-7) -> np.ndarray:
    _vector(residuals, "residuals")
    epsilon_value = _nonnegative(epsilon, "epsilon")
    tol = _positive(tolerance, "tolerance")
    distance = np.abs(residuals)
    result = np.full(residuals.shape, "inside_tube", dtype="<U13")
    result[np.abs(distance - epsilon_value) <= tol] = "on_tube"
    result[distance > epsilon_value + tol] = "outside_tube"
    return result


def dual_objective(beta: np.ndarray, y: np.ndarray, gram: np.ndarray, epsilon: float) -> float:
    _vector(beta, "beta")
    _vector(y, "y")
    epsilon_value = _nonnegative(epsilon, "epsilon")
    if beta.shape != y.shape or not _finite_array(gram) or gram.shape != (beta.size, beta.size):
        raise ValueError("beta、y和gram形状不一致")
    return float(-0.5 * beta @ gram @ beta - epsilon_value * np.sum(np.abs(beta)) + y @ beta)


def _pair_candidates(beta: np.ndarray, i: int, j: int, y: np.ndarray, K: np.ndarray, C: float, epsilon: float) -> list[float]:
    total = float(beta[i] + beta[j])
    low, high = max(-C, total - C), min(C, total + C)
    candidates = [low, high]
    if low <= 0.0 <= high:
        candidates.append(0.0)
    if low <= total <= high:
        candidates.append(total)
    base = beta.copy()
    base[i], base[j] = total, 0.0
    direction = np.zeros(beta.size)
    direction[i], direction[j] = -1.0, 1.0
    curvature = float(direction @ K @ direction)
    if curvature > 1e-14:
        smooth_linear = float(y[j] - y[i] - direction @ K @ base)
        for sign_i in (-1.0, 1.0):
            for sign_j in (-1.0, 1.0):
                value = (smooth_linear + epsilon * (sign_i - sign_j)) / curvature
                bi = total - value
                valid_signs = (bi >= -1e-12 if sign_i > 0 else bi <= 1e-12) and (value >= -1e-12 if sign_j > 0 else value <= 1e-12)
                if low - 1e-12 <= value <= high + 1e-12 and valid_signs:
                    candidates.append(float(np.clip(value, low, high)))
    return sorted(set(candidates))


def _recover_bias(beta: np.ndarray, y: np.ndarray, K: np.ndarray, C: float, epsilon: float, tolerance: float) -> float:
    raw = K @ beta
    free_positive = (beta > tolerance) & (beta < C - tolerance)
    free_negative = (beta < -tolerance) & (beta > -C + tolerance)
    candidates = np.concatenate((y[free_positive] - raw[free_positive] - epsilon, y[free_negative] - raw[free_negative] + epsilon))
    if candidates.size:
        return float(np.mean(candidates))
    lower, upper = -np.inf, np.inf
    for coefficient, target, value in zip(beta, y, raw):
        if coefficient >= C - tolerance:
            upper = min(upper, target - value - epsilon)
        elif coefficient <= -C + tolerance:
            lower = max(lower, target - value + epsilon)
        else:
            lower = max(lower, target - value - epsilon)
            upper = min(upper, target - value + epsilon)
    if np.isfinite(lower) and np.isfinite(upper):
        return float(0.5 * (lower + upper))
    if np.isfinite(lower):
        return float(lower)
    if np.isfinite(upper):
        return float(upper)
    return float(np.median(y - raw))


def fit_epsilon_svr(
    X: np.ndarray, y: np.ndarray, *, C: float = 1.0, epsilon: float = 0.1,
    kernel: str = "linear", gamma: float | None = None, tolerance: float = 1e-7,
    max_passes: int = 100,
) -> dict[str, object]:
    _matrix(X, "X")
    _vector(y, "y")
    if y.shape != (X.shape[0],):
        raise ValueError("y必须是形状(n,)的回归目标")
    C_value, epsilon_value, tol = _positive(C, "C"), _nonnegative(epsilon, "epsilon"), _positive(tolerance, "tolerance")
    pass_limit = _positive_int(max_passes, "max_passes")
    if kernel not in {"linear", "rbf"}:
        raise ValueError("kernel必须是linear或rbf")
    gamma_value = 1.0 / X.shape[1] if gamma is None else _positive(gamma, "gamma")
    Xf, yf = X.astype(float, copy=True), y.astype(float, copy=True)
    K = kernel_matrix(Xf, Xf, kernel=kernel, gamma=gamma_value)
    beta = np.zeros(X.shape[0])
    objective = dual_objective(beta, yf, K, epsilon_value)
    history, updates = [objective], 0
    for pass_index in range(1, pass_limit + 1):
        changed = 0
        for i in range(X.shape[0] - 1):
            for j in range(i + 1, X.shape[0]):
                best_value, best_t = objective, float(beta[j])
                total = float(beta[i] + beta[j])
                for t in _pair_candidates(beta, i, j, yf, K, C_value, epsilon_value):
                    candidate = beta.copy()
                    candidate[j], candidate[i] = t, total - t
                    value = dual_objective(candidate, yf, K, epsilon_value)
                    if value > best_value + tol or (abs(value - best_value) <= tol and t < best_t):
                        best_value, best_t = value, t
                if best_value > objective + tol:
                    beta[j], beta[i] = best_t, total - best_t
                    objective, changed, updates = best_value, changed + 1, updates + 1
        history.append(objective)
        if changed == 0:
            break
    bias = _recover_bias(beta, yf, K, C_value, epsilon_value, max(tol, 1e-8))
    return {"X_train": Xf, "y_train": yf, "beta": beta, "bias": bias, "C": C_value,
            "epsilon": epsilon_value, "kernel": kernel, "gamma": gamma_value,
            "passes": pass_index, "updates": updates, "dual_history": np.array(history)}


def _validate_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("model键集合无效")
    _matrix(model["X_train"], "model X_train")
    _vector(model["y_train"], "model y_train")
    _vector(model["beta"], "model beta")
    n = model["X_train"].shape[0]
    C = _positive(model["C"], "model C")
    if model["y_train"].shape != (n,) or model["beta"].shape != (n,) or np.any(np.abs(model["beta"]) > C + 1e-8) or not np.isfinite(model["bias"]):
        raise ValueError("model参数无效")


def decision_function(model: dict[str, object], X: np.ndarray, *, support_only: bool = True) -> np.ndarray:
    _validate_model(model)
    _matrix(X, "X")
    if X.shape[1] != model["X_train"].shape[1]:
        raise ValueError("X特征数与训练数据不一致")
    indices = np.flatnonzero(np.abs(model["beta"]) > 1e-10) if support_only else np.arange(model["beta"].size)
    if indices.size == 0:
        return np.full(X.shape[0], float(model["bias"]))
    K = kernel_matrix(X, model["X_train"][indices], kernel=model["kernel"], gamma=model["gamma"])
    return K @ model["beta"][indices] + float(model["bias"])


def support_vector_indices(model: dict[str, object]) -> np.ndarray:
    _validate_model(model)
    return np.flatnonzero(np.abs(model["beta"]) > 1e-10)


def kkt_residuals(model: dict[str, object], *, tolerance: float = 1e-7) -> np.ndarray:
    _validate_model(model)
    tol = _positive(tolerance, "tolerance")
    residual = model["y_train"] - decision_function(model, model["X_train"])
    beta, C, epsilon = model["beta"], float(model["C"]), float(model["epsilon"])
    result = np.empty_like(beta)
    zero = np.abs(beta) <= tol
    positive_free = (beta > tol) & (beta < C - tol)
    negative_free = (beta < -tol) & (beta > -C + tol)
    positive_bound = beta >= C - tol
    negative_bound = beta <= -C + tol
    result[zero] = np.maximum(0.0, np.abs(residual[zero]) - epsilon)
    result[positive_free] = np.abs(residual[positive_free] - epsilon)
    result[negative_free] = np.abs(residual[negative_free] + epsilon)
    result[positive_bound] = np.maximum(0.0, epsilon - residual[positive_bound])
    result[negative_bound] = np.maximum(0.0, residual[negative_bound] + epsilon)
    return result
