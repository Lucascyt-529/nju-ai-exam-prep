"""参考实现：NumPy二分类LDA。"""

import numpy as np


def _validate_data(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y必须是一维且样本数与X一致")
    if not np.all(np.isfinite(X)) or not np.all((y == 0) | (y == 1)):
        raise ValueError("X必须有限且y只能包含0/1")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("训练数据必须同时包含0类和1类")


def _validate_weights(X: np.ndarray, weights: np.ndarray) -> None:
    if not isinstance(weights, np.ndarray) or weights.ndim != 1 or weights.shape[0] != X.shape[1]:
        raise ValueError("weights必须具有形状(n_features,)")
    if not np.all(np.isfinite(weights)) or np.linalg.norm(weights) == 0:
        raise ValueError("weights必须是非零有限向量")


def class_means(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_data(X, y)
    return X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)


def within_class_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    _validate_data(X, y)
    mean0, mean1 = class_means(X, y)
    centered0 = X[y == 0] - mean0
    centered1 = X[y == 1] - mean1
    return centered0.T @ centered0 + centered1.T @ centered1


def fit_binary_lda(
    X: np.ndarray, y: np.ndarray, *, regularization: float = 0.0
) -> tuple[np.ndarray, float]:
    """返回投影权重和投影中点阈值。"""
    _validate_data(X, y)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization必须是非负有限数值")
    mean0, mean1 = class_means(X, y)
    difference = mean1 - mean0
    if np.linalg.norm(difference) == 0:
        raise ValueError("两类均值相同，无法得到LDA均值差方向")
    scatter = within_class_scatter(X, y)
    adjusted = scatter + regularization * np.eye(X.shape[1])
    weights = np.linalg.pinv(adjusted) @ difference
    if not np.all(np.isfinite(weights)) or np.linalg.norm(weights) == 0:
        raise ValueError("无法得到有效LDA投影方向")
    threshold = float(0.5 * (mean0 @ weights + mean1 @ weights))
    return weights, threshold


def project(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须只包含有限数值")
    _validate_weights(X, weights)
    return X @ weights


def predict_lda(
    X: np.ndarray, weights: np.ndarray, threshold: float
) -> np.ndarray:
    if not np.isfinite(threshold):
        raise ValueError("threshold必须是有限标量")
    return (project(X, weights) >= threshold).astype(int)


def fisher_ratio(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    _validate_data(X, y)
    _validate_weights(X, weights)
    projections = project(X, weights)
    projected0 = projections[y == 0]
    projected1 = projections[y == 1]
    numerator = float((projected1.mean() - projected0.mean()) ** 2)
    denominator = float(
        np.sum((projected0 - projected0.mean()) ** 2)
        + np.sum((projected1 - projected1.mean()) ** 2)
    )
    if denominator == 0:
        return np.inf if numerator > 0 else 0.0
    return numerator / denominator
