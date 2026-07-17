"""学生练习：一维网格SOM的竞争与邻域更新。"""

import numpy as np


def best_matching_unit(sample: np.ndarray, prototypes: np.ndarray) -> int:
    raise NotImplementedError("请完成 best_matching_unit")


def gaussian_neighborhood(winner: int, n_neurons: int, radius: float) -> np.ndarray:
    raise NotImplementedError("请完成 gaussian_neighborhood")


def update_prototypes(
    prototypes: np.ndarray,
    sample: np.ndarray,
    winner: int,
    *,
    learning_rate: float,
    radius: float,
) -> np.ndarray:
    raise NotImplementedError("请完成 update_prototypes")


def train_som(
    X: np.ndarray,
    *,
    n_neurons: int,
    epochs: int,
    initial_learning_rate: float = 0.5,
    final_learning_rate: float = 0.05,
    initial_radius: float | None = None,
    final_radius: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    raise NotImplementedError("请完成 train_som")


def map_samples(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 map_samples")
