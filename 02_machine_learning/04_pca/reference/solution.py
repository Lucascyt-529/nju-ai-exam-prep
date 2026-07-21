"""参考实现：协方差特征分解形式的 PCA。"""

import numpy as np


def _matrix(X: np.ndarray, name: str) -> None:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or 0 in X.shape
        or not np.issubdtype(X.dtype, np.number)
        or not np.all(np.isfinite(X))
    ):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _projection_parameters(mean: np.ndarray, components: np.ndarray) -> None:
    if (
        not isinstance(mean, np.ndarray)
        or mean.ndim != 1
        or mean.size == 0
        or not np.issubdtype(mean.dtype, np.number)
        or not np.all(np.isfinite(mean))
    ):
        raise ValueError("mean必须是一维有限数值数组")
    _matrix(components, "components")
    if components.shape[1] != mean.size:
        raise ValueError("components的特征数必须与mean长度一致")


def _orient_components(components: np.ndarray) -> np.ndarray:
    oriented = components.copy()
    for row in range(oriented.shape[0]):
        pivot = int(np.argmax(np.abs(oriented[row])))
        if oriented[row, pivot] < 0:
            oriented[row] *= -1
    return oriented


def fit_pca(X: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    _matrix(X, "X")
    n_features = X.shape[1]
    if (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
        or n_components <= 0
        or n_components > n_features
    ):
        raise ValueError("n_components必须是1到特征数之间的整数")

    values = X.astype(float)
    mean = np.mean(values, axis=0)
    centered = values - mean
    denominator = X.shape[0] - 1
    if denominator == 0:
        covariance = np.zeros((n_features, n_features), dtype=float)
    else:
        covariance = centered.T @ centered / denominator

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    all_variance = np.maximum(eigenvalues[order], 0.0)
    all_components = _orient_components(eigenvectors[:, order].T)
    selected_variance = all_variance[: int(n_components)]
    components = all_components[: int(n_components)]
    total_variance = float(np.sum(all_variance))
    if total_variance == 0.0:
        explained_ratio = np.zeros(int(n_components), dtype=float)
    else:
        explained_ratio = selected_variance / total_variance

    return {
        "mean": mean,
        "components": components,
        "explained_variance": selected_variance,
        "explained_variance_ratio": explained_ratio,
        "all_explained_variance": all_variance,
        "all_components": all_components,
        "covariance": covariance,
    }


def transform_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    _matrix(X, "X")
    _projection_parameters(mean, components)
    if X.shape[1] != mean.size:
        raise ValueError("X的特征数必须与mean长度一致")
    return (X.astype(float) - mean.astype(float)) @ components.astype(float).T


def inverse_transform_pca(Z: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    _matrix(Z, "Z")
    _projection_parameters(mean, components)
    if Z.shape[1] != components.shape[0]:
        raise ValueError("Z的列数必须等于主成分数")
    return Z.astype(float) @ components.astype(float) + mean.astype(float)


def reconstruction_mse(X: np.ndarray, reconstructed: np.ndarray) -> float:
    _matrix(X, "X")
    _matrix(reconstructed, "reconstructed")
    if X.shape != reconstructed.shape:
        raise ValueError("X与reconstructed形状必须相同")
    return float(np.mean((X.astype(float) - reconstructed.astype(float)) ** 2))
