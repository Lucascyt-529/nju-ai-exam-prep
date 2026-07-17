"""学生练习：聚类距离与性能度量。"""

import numpy as np


def pairwise_minkowski(X: np.ndarray, Z: np.ndarray, *, p: float = 2.0) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_minkowski")


def within_cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("请完成 within_cluster_sse")


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 silhouette_samples")


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    raise NotImplementedError("请完成 adjusted_rand_index")
