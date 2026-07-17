"""学生练习：K-means++、分配更新与空簇处理。"""

import numpy as np


def kmeans_plus_plus(X: np.ndarray, n_clusters: int, *, random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 kmeans_plus_plus")


def assign_labels(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 assign_labels")


def fit_kmeans(X: np.ndarray, n_clusters: int, *, random_state: int = 0, max_iterations: int = 100, tolerance: float = 1e-6) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_kmeans")
