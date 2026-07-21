"""学生练习：PCA 中心化、主轴、投影与重构。"""

import numpy as np


def fit_pca(X: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 fit_pca")


def transform_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 transform_pca")


def inverse_transform_pca(Z: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 inverse_transform_pca")


def reconstruction_mse(X: np.ndarray, reconstructed: np.ndarray) -> float:
    raise NotImplementedError("请完成 reconstruction_mse")
