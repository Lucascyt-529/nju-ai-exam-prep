"""学生练习：LVQ带标签原型学习。"""

import numpy as np


def nearest_prototype(X: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 nearest_prototype")


def lvq_update(prototypes: np.ndarray, prototype_labels: np.ndarray, x: np.ndarray, y: object, learning_rate: float) -> tuple[np.ndarray, int, bool]:
    raise NotImplementedError("请完成 lvq_update")


def fit_lvq(X: np.ndarray, y: np.ndarray, *, prototypes_per_class: int = 1, epochs: int = 20, learning_rate: float = 0.2, decay: float = 0.95, random_state: int = 0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_lvq")
