"""参考实现：经典 MDS 的距离双中心化与谱嵌入。"""

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


def _distance_matrix(distance_matrix: np.ndarray) -> None:
    _matrix(distance_matrix, "distance_matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix必须是方阵")
    if np.any(distance_matrix < -1e-12):
        raise ValueError("距离不能为负")
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("distance_matrix必须对称")
    if not np.allclose(np.diag(distance_matrix), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("distance_matrix对角线必须为0")


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    _matrix(X, "X")
    difference = X.astype(float)[:, None, :] - X.astype(float)[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def double_center(distance_matrix: np.ndarray) -> np.ndarray:
    """把距离矩阵转换为中心化的样本内积矩阵。"""
    _distance_matrix(distance_matrix)
    n_samples = distance_matrix.shape[0]
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    squared = np.maximum(distance_matrix.astype(float), 0.0) ** 2
    gram = -0.5 * centering @ squared @ centering
    return 0.5 * (gram + gram.T)


def _orient_columns(eigenvectors: np.ndarray) -> np.ndarray:
    oriented = eigenvectors.copy()
    for column in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[pivot, column] < 0:
            oriented[:, column] *= -1
    return oriented


def classical_mds(distance_matrix: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    _distance_matrix(distance_matrix)
    n_samples = distance_matrix.shape[0]
    if (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
        or n_components <= 0
        or n_components > n_samples
    ):
        raise ValueError("n_components必须是1到样本数之间的整数")

    gram = double_center(distance_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _orient_columns(eigenvectors[:, order])
    selected_values = np.maximum(eigenvalues[: int(n_components)], 0.0)
    coordinates = eigenvectors[:, : int(n_components)] * np.sqrt(selected_values)[None, :]
    return {
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "gram": gram,
        "selected_eigenvalues": selected_values,
    }


def normalized_stress(distance_matrix: np.ndarray, coordinates: np.ndarray) -> float:
    _distance_matrix(distance_matrix)
    _matrix(coordinates, "coordinates")
    if coordinates.shape[0] != distance_matrix.shape[0]:
        raise ValueError("坐标样本数必须与距离矩阵一致")
    embedded_distances = pairwise_euclidean(coordinates)
    upper = np.triu_indices(distance_matrix.shape[0], k=1)
    numerator = float(np.sum((embedded_distances[upper] - distance_matrix[upper]) ** 2))
    denominator = float(np.sum(distance_matrix[upper] ** 2))
    if denominator == 0.0:
        return 0.0 if numerator <= 1e-24 else float("inf")
    return float(np.sqrt(numerator / denominator))
