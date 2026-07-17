"""参考实现：只使用 NumPy 的 kNN 分类和回归。"""

import numpy as np


WEIGHTS = {"uniform", "distance"}


def _matrix(X: np.ndarray, name: str) -> None:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or 0 in X.shape
        or not np.issubdtype(X.dtype, np.number)
        or not np.all(np.isfinite(X))
    ):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _inputs(X_query: np.ndarray, X_train: np.ndarray, k: int) -> None:
    _matrix(X_query, "X_query")
    _matrix(X_train, "X_train")
    if X_query.shape[1] != X_train.shape[1]:
        raise ValueError("查询集与训练集的特征数必须一致")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0 or k > X_train.shape[0]:
        raise ValueError("k必须是1到训练样本数之间的整数")


def _targets(y_train: np.ndarray, n_train: int, *, numeric: bool) -> None:
    if not isinstance(y_train, np.ndarray) or y_train.shape != (n_train,):
        raise ValueError("y_train必须是形状(n_train,)的一维数组")
    if numeric:
        if not np.issubdtype(y_train.dtype, np.number) or not np.all(np.isfinite(y_train)):
            raise ValueError("回归目标必须是有限数值")
    elif y_train.dtype.kind in "fc" and not np.all(np.isfinite(y_train)):
        raise ValueError("分类标签不能包含非有限值")


def _weight_mode(weights: str) -> None:
    if not isinstance(weights, str) or weights not in WEIGHTS:
        raise ValueError("weights必须是uniform或distance")


def pairwise_euclidean(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """返回形状为 (n_query, n_train) 的距离矩阵。"""
    _matrix(X_query, "X_query")
    _matrix(X_train, "X_train")
    if X_query.shape[1] != X_train.shape[1]:
        raise ValueError("查询集与训练集的特征数必须一致")
    difference = X_query.astype(float)[:, None, :] - X_train.astype(float)[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def kneighbors(X_query: np.ndarray, X_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """返回邻居距离和下标；距离平局时训练下标小者优先。"""
    _inputs(X_query, X_train, k)
    distances = pairwise_euclidean(X_query, X_train)
    order = np.argsort(distances, axis=1, kind="stable")[:, : int(k)]
    selected = np.take_along_axis(distances, order, axis=1)
    return selected, order


def _row_weights(distances: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(distances, dtype=float)
    zero_mask = distances == 0.0
    if np.any(zero_mask):
        return zero_mask.astype(float)
    return 1.0 / distances


def predict_classification(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    *,
    weights: str = "uniform",
) -> np.ndarray:
    _inputs(X_query, X_train, k)
    _targets(y_train, X_train.shape[0], numeric=False)
    _weight_mode(weights)
    distances, indices = kneighbors(X_query, X_train, k)
    classes = np.unique(y_train)
    predictions = []
    for row_distances, row_indices in zip(distances, indices):
        neighbor_labels = y_train[row_indices]
        row_weights = _row_weights(row_distances, weights)
        scores = np.array([np.sum(row_weights[neighbor_labels == label]) for label in classes])
        predictions.append(classes[int(np.argmax(scores))])
    return np.asarray(predictions, dtype=y_train.dtype)


def predict_regression(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    *,
    weights: str = "uniform",
) -> np.ndarray:
    _inputs(X_query, X_train, k)
    _targets(y_train, X_train.shape[0], numeric=True)
    _weight_mode(weights)
    distances, indices = kneighbors(X_query, X_train, k)
    predictions = np.empty(X_query.shape[0], dtype=float)
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        row_weights = _row_weights(row_distances, weights)
        predictions[row] = np.average(y_train[row_indices].astype(float), weights=row_weights)
    return predictions
