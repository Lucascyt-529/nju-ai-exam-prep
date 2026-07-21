"""学生练习：加权决策树桩与AdaBoost。"""

import numpy as np


def fit_weighted_stump(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_weighted_stump")


def update_sample_weights(sample_weight: np.ndarray, y: np.ndarray, prediction: np.ndarray, alpha: float) -> np.ndarray:
    raise NotImplementedError("请完成 update_sample_weights")


def fit_adaboost(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_adaboost")


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict")
