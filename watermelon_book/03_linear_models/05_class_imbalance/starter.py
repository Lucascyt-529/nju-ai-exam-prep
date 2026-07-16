"""学生练习：类别不平衡的采样、加权和决策调整。"""

import numpy as np


def class_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 class_counts")


def balanced_class_weights(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 balanced_class_weights")


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 balanced_sample_weights")


def random_undersample_indices(y: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    raise NotImplementedError("请完成 random_undersample_indices")


def random_oversample_indices(y: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    raise NotImplementedError("请完成 random_oversample_indices")


def cost_sensitive_threshold(
    false_positive_cost: float, false_negative_cost: float
) -> float:
    raise NotImplementedError("请完成 cost_sensitive_threshold")


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    raise NotImplementedError("请完成 predict_with_threshold")


def correct_prior_shift(
    probabilities: np.ndarray,
    *,
    source_positive_prior: float,
    target_positive_prior: float,
) -> np.ndarray:
    raise NotImplementedError("请完成 correct_prior_shift")


def prior_shift_decision_threshold(
    *, source_positive_prior: float, target_positive_prior: float
) -> float:
    raise NotImplementedError("请完成 prior_shift_decision_threshold")
