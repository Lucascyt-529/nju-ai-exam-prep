"""学生练习：经典 MDS 的双中心化与特征分解。"""

import numpy as np


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 pairwise_euclidean")


def double_center(distance_matrix: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 double_center")


def classical_mds(distance_matrix: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 classical_mds")


def normalized_stress(distance_matrix: np.ndarray, coordinates: np.ndarray) -> float:
    raise NotImplementedError("请完成 normalized_stress")
