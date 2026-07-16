"""学生练习：NumPy实现OvR与OvO多分类拆分。"""

import numpy as np


def fit_ovr(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回classes、weights和biases。"""
    raise NotImplementedError("请完成 fit_ovr")


def decision_function_ovr(
    X: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function_ovr")


def predict_ovr(
    X: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_ovr")


def fit_ovo(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回classes、class_pairs、weights和biases。"""
    raise NotImplementedError("请完成 fit_ovo")


def decision_function_ovo(
    X: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function_ovo")


def ovo_vote_counts(
    pair_scores: np.ndarray, class_pairs: np.ndarray, n_classes: int
) -> np.ndarray:
    raise NotImplementedError("请完成 ovo_vote_counts")


def predict_ovo(
    X: np.ndarray,
    classes: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_ovo")
