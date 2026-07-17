"""学生练习：NumPy实现OvR、OvO与ECOC多分类拆分。"""

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


def make_ovr_coding_matrix(n_classes: int) -> np.ndarray:
    raise NotImplementedError("请完成 make_ovr_coding_matrix")


def make_ovo_coding_matrix(n_classes: int) -> np.ndarray:
    raise NotImplementedError("请完成 make_ovo_coding_matrix")


def ecoc_training_targets(
    y: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    classifier_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 ecoc_training_targets")


def fit_ecoc(
    X: np.ndarray, y: np.ndarray, coding_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 fit_ecoc")


def decision_function_ecoc(
    X: np.ndarray,
    coding_matrix: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function_ecoc")


def hard_ecoc_codes(scores: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 hard_ecoc_codes")


def ecoc_distances(
    predicted_codes: np.ndarray,
    coding_matrix: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    raise NotImplementedError("请完成 ecoc_distances")


def decode_ecoc(
    predicted_codes: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    raise NotImplementedError("请完成 decode_ecoc")


def predict_ecoc(
    X: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_ecoc")


def minimum_hamming_distance(coding_matrix: np.ndarray) -> int:
    raise NotImplementedError("请完成 minimum_hamming_distance")


def binary_error_correction_capacity(coding_matrix: np.ndarray) -> int:
    raise NotImplementedError("请完成 binary_error_correction_capacity")
