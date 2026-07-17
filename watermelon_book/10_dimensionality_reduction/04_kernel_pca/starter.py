"""学生练习：核矩阵、中心化和核 PCA。"""

import numpy as np


def kernel_matrix(X: np.ndarray, Z: np.ndarray, *, kernel: str = "rbf", degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    raise NotImplementedError("请完成 kernel_matrix")


def center_train_kernel(K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    raise NotImplementedError("请完成 center_train_kernel")


def fit_kernel_pca(X: np.ndarray, n_components: int, **kernel_options: object) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_kernel_pca")


def transform_kernel_pca(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 transform_kernel_pca")
