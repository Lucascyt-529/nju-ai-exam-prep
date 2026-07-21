"""学生练习：Bootstrap、Bagging与OOB评估。"""

import numpy as np


def bootstrap_sample_indices(n_samples: int, n_estimators: int, *, random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 bootstrap_sample_indices")


def out_of_bag_indices(indices: np.ndarray, n_samples: int) -> np.ndarray:
    raise NotImplementedError("请完成 out_of_bag_indices")


def fit_bagging_stumps(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20, random_state: int = 0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_bagging_stumps")


def oob_decision_function(model: dict[str, object], X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 oob_decision_function")
