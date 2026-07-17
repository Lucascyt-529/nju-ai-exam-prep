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


def pairwise_weighted_minkowski(X: np.ndarray, Z: np.ndarray, *, p: float = 2.0,
                                weights: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_weighted_minkowski")


def fit_vdm(categorical_data: np.ndarray, group_labels: np.ndarray, *,
            alpha: float = 0.0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_vdm")


def pairwise_vdm(X: np.ndarray, Z: np.ndarray, model: dict[str, object], *,
                 p: float = 1.0, weights: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_vdm")


def pairwise_mixed_distance(
    X_numeric: np.ndarray, Z_numeric: np.ndarray,
    X_categorical: np.ndarray, Z_categorical: np.ndarray,
    vdm_model: dict[str, object], *, p: float = 2.0,
    numeric_weights: np.ndarray | None = None,
    categorical_weights: np.ndarray | None = None,
) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_mixed_distance")
