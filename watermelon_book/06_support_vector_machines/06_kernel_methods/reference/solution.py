"""参考实现：核矩阵、表示定理预测形状与二分类KLDA。"""

from numbers import Real
import numpy as np


def _finite_matrix(X: np.ndarray, name: str) -> None:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X))):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _kernel_options(kernel: str, gamma: float, degree: int, coef0: float) -> None:
    if kernel not in {"linear", "polynomial", "rbf"}:
        raise ValueError("kernel必须是linear、polynomial或rbf")
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real)
            or not np.isfinite(gamma) or gamma <= 0):
        raise ValueError("gamma必须是有限正数")
    if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
            or degree <= 0):
        raise ValueError("degree必须是正整数")
    if isinstance(coef0, (bool, np.bool_)) or not isinstance(coef0, Real) or not np.isfinite(coef0):
        raise ValueError("coef0必须是有限数")


def kernel_matrix(X: np.ndarray, Z: np.ndarray, *, kernel: str = "linear",
                  gamma: float = 1.0, degree: int = 2,
                  coef0: float = 1.0) -> np.ndarray:
    _finite_matrix(X, "X"); _finite_matrix(Z, "Z"); _kernel_options(kernel, gamma, degree, coef0)
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X与Z特征数不一致")
    X_float = X.astype(float, copy=False); Z_float = Z.astype(float, copy=False)
    dot = X_float @ Z_float.T
    if kernel == "linear":
        return dot
    if kernel == "polynomial":
        return (float(gamma) * dot + float(coef0)) ** int(degree)
    distances_squared = (np.sum(X_float ** 2, axis=1)[:, None]
                         + np.sum(Z_float ** 2, axis=1)[None, :] - 2.0 * dot)
    return np.exp(-float(gamma) * np.maximum(distances_squared, 0.0))


def representer_prediction(K_query_train: np.ndarray, coefficients: np.ndarray,
                           *, bias: float = 0.0) -> np.ndarray:
    _finite_matrix(K_query_train, "K_query_train")
    if (not isinstance(coefficients, np.ndarray) or coefficients.ndim != 1
            or len(coefficients) != K_query_train.shape[1]
            or not np.issubdtype(coefficients.dtype, np.number)
            or not np.all(np.isfinite(coefficients))):
        raise ValueError("coefficients必须是与核矩阵列数匹配的有限一维数组")
    if isinstance(bias, (bool, np.bool_)) or not isinstance(bias, Real) or not np.isfinite(bias):
        raise ValueError("bias必须是有限数")
    return K_query_train @ coefficients.astype(float, copy=False) + float(bias)


def _binary_labels(y: np.ndarray, n_samples: int) -> np.ndarray:
    if (not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples
            or not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y))):
        raise ValueError("y必须是长度匹配的有限数值一维数组")
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("KLDA当前只支持恰好两个类别")
    return classes


def kernel_scatter_matrices(K: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _finite_matrix(K, "K")
    if K.shape[0] != K.shape[1] or not np.allclose(K, K.T, atol=1e-10):
        raise ValueError("K必须是对称方阵")
    classes = _binary_labels(y, len(K))
    class_indices = [np.flatnonzero(y == label) for label in classes]
    kernel_means = [np.mean(K[:, indices], axis=1) for indices in class_indices]
    difference = kernel_means[1] - kernel_means[0]
    between = np.outer(difference, difference)
    within = np.zeros_like(K, dtype=float)
    for indices in class_indices:
        class_kernel = K[:, indices]
        centered = class_kernel - np.mean(class_kernel, axis=1, keepdims=True)
        within += centered @ centered.T
    return between, within


def fit_kernel_lda(X: np.ndarray, y: np.ndarray, *, kernel: str = "linear",
                   gamma: float = 1.0, degree: int = 2,
                   coef0: float = 1.0,
                   regularization: float = 1e-6) -> dict[str, object]:
    _finite_matrix(X, "X"); classes = _binary_labels(y, len(X))
    _kernel_options(kernel, gamma, degree, coef0)
    if (isinstance(regularization, (bool, np.bool_)) or not isinstance(regularization, Real)
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("regularization必须是有限非负数")
    K = kernel_matrix(X, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    between, within = kernel_scatter_matrices(K, y)
    difference = np.mean(K[:, y == classes[1]], axis=1) - np.mean(K[:, y == classes[0]], axis=1)
    coefficients = np.linalg.pinv(within + float(regularization) * np.eye(len(X))) @ difference
    norm = np.linalg.norm(coefficients)
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError("两类在所选核空间中无法形成有效判别方向")
    coefficients = coefficients / norm
    projections = K @ coefficients
    class_means = np.array([np.mean(projections[y == label]) for label in classes])
    if class_means[1] < class_means[0]:
        coefficients = -coefficients; projections = -projections; class_means = -class_means
    threshold = float(np.mean(class_means))
    return {"X_train": X.astype(float, copy=True), "classes": classes.copy(),
            "coefficients": coefficients, "threshold": threshold,
            "kernel": kernel, "gamma": float(gamma), "degree": int(degree),
            "coef0": float(coef0), "regularization": float(regularization),
            "between_scatter": between, "within_scatter": within,
            "train_projections": projections}


def _validate_model(model: dict[str, object]) -> None:
    required = {"X_train", "classes", "coefficients", "threshold", "kernel", "gamma", "degree",
                "coef0", "regularization", "between_scatter", "within_scatter", "train_projections"}
    if not isinstance(model, dict) or set(model) != required:
        raise ValueError("model键集合无效")


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model); _finite_matrix(X, "X")
    K_query = kernel_matrix(X, model["X_train"], kernel=model["kernel"], gamma=model["gamma"],
                            degree=model["degree"], coef0=model["coef0"])
    return representer_prediction(K_query, model["coefficients"], bias=-model["threshold"])


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    scores = decision_function(model, X)
    return np.where(scores >= 0, model["classes"][1], model["classes"][0])


def fisher_ratio(model: dict[str, object]) -> float:
    _validate_model(model)
    coefficients = model["coefficients"]
    numerator = float(coefficients @ model["between_scatter"] @ coefficients)
    denominator = float(coefficients @ model["within_scatter"] @ coefficients)
    return float(np.inf if denominator <= 1e-15 and numerator > 0 else numerator / denominator)
