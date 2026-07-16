"""参考实现：类别不平衡的采样、加权和决策调整。"""

import numpy as np


def _validate_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空一维数组")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须包含有限数值标签")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("y必须至少包含两个类别")
    return classes, counts


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities必须是非空一维数组")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities必须位于[0,1]")


def _validate_prior(value: float, name: str) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < float(value) < 1:
        raise ValueError(f"{name}必须是(0,1)内的有限数值")
    return float(value)


def class_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回升序类别标签和对应样本数。"""
    return _validate_labels(y)


def balanced_class_weights(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = _validate_labels(y)
    weights = y.size / (classes.size * counts.astype(float))
    return classes, weights


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, weights = balanced_class_weights(y)
    class_positions = np.searchsorted(classes, y)
    return weights[class_positions]


def random_undersample_indices(y: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    classes, counts = _validate_labels(y)
    rng = np.random.default_rng(random_state)
    target_count = int(counts.min())
    selected_parts = [
        rng.choice(np.flatnonzero(y == label), size=target_count, replace=False)
        for label in classes
    ]
    selected = np.concatenate(selected_parts)
    return rng.permutation(selected)


def random_oversample_indices(y: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    classes, counts = _validate_labels(y)
    rng = np.random.default_rng(random_state)
    target_count = int(counts.max())
    selected_parts = []
    for label in classes:
        original = np.flatnonzero(y == label)
        extra = rng.choice(original, size=target_count - original.size, replace=True)
        selected_parts.append(np.concatenate((original, extra)))
    selected = np.concatenate(selected_parts)
    return rng.permutation(selected)


def cost_sensitive_threshold(
    false_positive_cost: float, false_negative_cost: float
) -> float:
    values = (false_positive_cost, false_negative_cost)
    if any(not np.isscalar(value) or not np.isfinite(value) or float(value) <= 0 for value in values):
        raise ValueError("两种误分类代价都必须是正的有限数值")
    fp_cost = float(false_positive_cost)
    fn_cost = float(false_negative_cost)
    return fp_cost / (fp_cost + fn_cost)


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    _validate_probabilities(probabilities)
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold必须是[0,1]内的有限数值")
    return (probabilities >= float(threshold)).astype(int)


def correct_prior_shift(
    probabilities: np.ndarray,
    *,
    source_positive_prior: float,
    target_positive_prior: float,
) -> np.ndarray:
    _validate_probabilities(probabilities)
    source = _validate_prior(source_positive_prior, "source_positive_prior")
    target = _validate_prior(target_positive_prior, "target_positive_prior")
    positive_part = probabilities * (target / source)
    negative_part = (1.0 - probabilities) * ((1.0 - target) / (1.0 - source))
    return positive_part / (positive_part + negative_part)


def prior_shift_decision_threshold(
    *, source_positive_prior: float, target_positive_prior: float
) -> float:
    source = _validate_prior(source_positive_prior, "source_positive_prior")
    target = _validate_prior(target_positive_prior, "target_positive_prior")
    odds_ratio = (target / (1.0 - target)) / (source / (1.0 - source))
    return 1.0 / (1.0 + odds_ratio)
