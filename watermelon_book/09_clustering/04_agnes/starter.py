"""学生练习：AGNES凝聚层次聚类。"""

import numpy as np


def cluster_distance(cluster_a: tuple[int, ...], cluster_b: tuple[int, ...], sample_distances: np.ndarray, linkage: str) -> float:
    raise NotImplementedError("请完成 cluster_distance")


def fit_agnes(X: np.ndarray, n_clusters: int, *, linkage: str = "average") -> dict[str, object]:
    raise NotImplementedError("请完成 fit_agnes")
