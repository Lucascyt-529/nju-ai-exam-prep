"""学生练习：NumPy二分类与多分类LDA。"""

import numpy as np


def class_means(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 class_means")


def within_class_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 within_class_scatter")


def fit_binary_lda(
    X: np.ndarray, y: np.ndarray, *, regularization: float = 0.0
) -> tuple[np.ndarray, float]:
    """返回投影权重和投影中点阈值。"""
    raise NotImplementedError("请完成 fit_binary_lda")


def project(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 project")


def predict_lda(
    X: np.ndarray, weights: np.ndarray, threshold: float
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_lda")


def fisher_ratio(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    raise NotImplementedError("请完成 fisher_ratio")


def multiclass_scatter_matrices(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 multiclass_scatter_matrices")


def fit_multiclass_lda(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int | None = None,
    regularization: float = 1e-8,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 fit_multiclass_lda")


def project_multiclass(X: np.ndarray, projection: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 project_multiclass")


def predict_multiclass_lda(
    X: np.ndarray,
    classes: np.ndarray,
    projection: np.ndarray,
    projected_centroids: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_multiclass_lda")


def multiclass_trace_ratio(
    projection: np.ndarray,
    within: np.ndarray,
    between: np.ndarray,
) -> float:
    raise NotImplementedError("请完成 multiclass_trace_ratio")
