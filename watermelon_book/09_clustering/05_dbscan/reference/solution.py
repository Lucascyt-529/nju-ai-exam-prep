"""参考实现：只使用 NumPy 的确定性 DBSCAN。"""

from collections import deque
from numbers import Real

import numpy as np


def _validate_X(X: np.ndarray) -> None:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or 0 in X.shape
        or not np.issubdtype(X.dtype, np.number)
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X 必须是非空、有限数值的二维 NumPy 数组")


def _validate_parameters(eps: float, min_samples: int) -> None:
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps 必须是有限正数")
    if (
        isinstance(min_samples, (bool, np.bool_))
        or not isinstance(min_samples, (int, np.integer))
        or min_samples <= 0
    ):
        raise ValueError("min_samples 必须是正整数")


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """返回形状为 (n, n) 的欧氏距离矩阵。"""
    _validate_X(X)
    values = X.astype(float, copy=False)
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def epsilon_neighborhoods(X: np.ndarray, eps: float) -> np.ndarray:
    """返回包含自身、使用闭区间 distance <= eps 的邻域矩阵。"""
    _validate_X(X)
    _validate_parameters(eps, 1)
    return pairwise_euclidean(X) <= float(eps)


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict[str, object]:
    """按样本下标顺序进行确定性的密度簇扩展。"""
    _validate_X(X)
    _validate_parameters(eps, min_samples)

    neighborhood_matrix = epsilon_neighborhoods(X, eps)
    neighbor_counts = np.sum(neighborhood_matrix, axis=1)
    core_mask = neighbor_counts >= int(min_samples)
    neighborhoods = tuple(np.flatnonzero(row) for row in neighborhood_matrix)

    labels = np.full(X.shape[0], -1, dtype=int)
    expanded_core = np.zeros(X.shape[0], dtype=bool)
    cluster_id = 0

    for seed in range(X.shape[0]):
        if not core_mask[seed] or expanded_core[seed]:
            continue

        labels[seed] = cluster_id
        queue = deque([seed])
        queued = np.zeros(X.shape[0], dtype=bool)
        queued[seed] = True

        while queue:
            current = queue.popleft()
            expanded_core[current] = True

            for neighbor in neighborhoods[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                if core_mask[neighbor] and not expanded_core[neighbor] and not queued[neighbor]:
                    queue.append(int(neighbor))
                    queued[neighbor] = True

        cluster_id += 1

    border_mask = (labels >= 0) & ~core_mask
    noise_mask = labels == -1
    clusters = tuple(tuple(np.flatnonzero(labels == label)) for label in range(cluster_id))

    return {
        "labels": labels,
        "core_mask": core_mask,
        "border_mask": border_mask,
        "noise_mask": noise_mask,
        "neighbor_counts": neighbor_counts,
        "neighborhood_matrix": neighborhood_matrix,
        "neighborhoods": neighborhoods,
        "clusters": clusters,
        "eps": float(eps),
        "min_samples": int(min_samples),
    }
