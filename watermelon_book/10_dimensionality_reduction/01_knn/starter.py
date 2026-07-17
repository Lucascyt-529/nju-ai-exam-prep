"""学生练习：kNN 邻居选择、分类和回归。"""

import numpy as np


def pairwise_euclidean(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_euclidean")


def kneighbors(X_query: np.ndarray, X_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 kneighbors")


def predict_classification(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    *,
    weights: str = "uniform",
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_classification")


def predict_regression(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    *,
    weights: str = "uniform",
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_regression")
