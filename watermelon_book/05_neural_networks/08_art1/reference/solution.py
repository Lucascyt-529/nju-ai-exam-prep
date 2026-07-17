"""参考实现：二值ART1的选择、警戒与快速交集学习。"""

import numpy as np


def _is_binary_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
        and np.all(np.isin(value, [0, 1]))
    )


def _validate_patterns(X: np.ndarray) -> None:
    if (
        not _is_binary_array(X)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or np.any(np.sum(X, axis=1) == 0)
    ):
        raise ValueError("X必须是非空二维0/1数组且每行至少包含一个1")


def _validate_pattern(pattern: np.ndarray, n_features: int) -> None:
    if (
        not _is_binary_array(pattern)
        or pattern.shape != (n_features,)
        or np.sum(pattern) == 0
    ):
        raise ValueError("pattern必须是匹配特征数的非零一维0/1数组")


def _validate_prototypes(prototypes: np.ndarray) -> None:
    if (
        not _is_binary_array(prototypes)
        or prototypes.ndim != 2
        or prototypes.shape[0] == 0
        or prototypes.shape[1] == 0
    ):
        raise ValueError("prototypes必须是非空二维0/1数组")


def _positive_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _vigilance(value: object) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and 0 <= value <= 1
    )
    if not valid:
        raise ValueError("vigilance必须位于0到1")
    return float(value)


def choice_scores(pattern: np.ndarray, prototypes: np.ndarray, *, alpha: float) -> np.ndarray:
    _validate_prototypes(prototypes)
    _validate_pattern(pattern, prototypes.shape[1])
    alpha_value = _positive_scalar(alpha, "alpha")
    intersection = np.logical_and(prototypes, pattern[None, :]).sum(axis=1)
    prototype_sizes = prototypes.sum(axis=1)
    return intersection / (alpha_value + prototype_sizes)


def match_ratio(pattern: np.ndarray, prototype: np.ndarray) -> float:
    if not _is_binary_array(prototype) or prototype.ndim != 1 or prototype.size == 0:
        raise ValueError("prototype必须是一维0/1数组")
    _validate_pattern(pattern, prototype.size)
    intersection = np.logical_and(pattern, prototype).sum()
    return float(intersection / pattern.sum())


def select_resonant_category(
    pattern: np.ndarray,
    prototypes: np.ndarray,
    *,
    alpha: float,
    vigilance: float,
) -> int | None:
    scores = choice_scores(pattern, prototypes, alpha=alpha)
    vigilance_value = _vigilance(vigilance)
    order = np.argsort(-scores, kind="stable")
    for category in order:
        if match_ratio(pattern, prototypes[category]) >= vigilance_value:
            return int(category)
    return None


def train_art1(
    X: np.ndarray,
    *,
    alpha: float = 0.001,
    vigilance: float = 0.8,
    max_categories: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    _validate_patterns(X)
    alpha_value = _positive_scalar(alpha, "alpha")
    vigilance_value = _vigilance(vigilance)
    if max_categories is not None and (
        not isinstance(max_categories, (int, np.integer))
        or isinstance(max_categories, (bool, np.bool_))
        or max_categories <= 0
    ):
        raise ValueError("max_categories必须是正整数或None")

    patterns = X.astype(int, copy=True)
    categories: list[np.ndarray] = []
    assignments: list[int] = []
    category_history: list[int] = []
    for pattern in patterns:
        winner = None
        if categories:
            winner = select_resonant_category(
                pattern,
                np.stack(categories),
                alpha=alpha_value,
                vigilance=vigilance_value,
            )
        if winner is None:
            if max_categories is not None and len(categories) >= max_categories:
                raise RuntimeError("没有类别通过警戒检验且已达到max_categories")
            categories.append(pattern.copy())
            winner = len(categories) - 1
        else:
            categories[winner] = np.logical_and(categories[winner], pattern).astype(int)
        assignments.append(winner)
        category_history.append(len(categories))
    return np.stack(categories), np.array(assignments, dtype=int), category_history


def predict_art1(
    X: np.ndarray,
    prototypes: np.ndarray,
    *,
    alpha: float = 0.001,
    vigilance: float = 0.8,
) -> np.ndarray:
    _validate_patterns(X)
    _validate_prototypes(prototypes)
    if X.shape[1] != prototypes.shape[1]:
        raise ValueError("X和prototypes特征数必须一致")
    alpha_value = _positive_scalar(alpha, "alpha")
    vigilance_value = _vigilance(vigilance)
    assignments = []
    for pattern in X:
        category = select_resonant_category(
            pattern,
            prototypes,
            alpha=alpha_value,
            vigilance=vigilance_value,
        )
        assignments.append(-1 if category is None else category)
    return np.array(assignments, dtype=int)
