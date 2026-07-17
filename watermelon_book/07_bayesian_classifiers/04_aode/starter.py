"""学生练习：AODE半朴素贝叶斯。"""

import numpy as np


def fit_aode(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, min_parent_count: int = 1) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_aode")


def eligible_parent_indices(model: dict[str, object], row: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 eligible_parent_indices")


def joint_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 joint_log_scores")


def predict_proba(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_proba")
