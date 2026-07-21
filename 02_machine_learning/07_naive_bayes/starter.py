"""学生练习：离散与高斯朴素贝叶斯。"""

import numpy as np


def fit_categorical_nb(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_categorical_nb")


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray, *, variance_floor: float = 1e-9) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_gaussian_nb")


def joint_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 joint_log_scores")


def predict_proba(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_proba")


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict")
