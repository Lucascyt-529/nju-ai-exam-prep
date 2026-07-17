"""学生练习：邻域、核心点与 DBSCAN 簇扩展。"""

import numpy as np


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_euclidean")


def epsilon_neighborhoods(X: np.ndarray, eps: float) -> np.ndarray:
    raise NotImplementedError("请完成 epsilon_neighborhoods")


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_dbscan")
