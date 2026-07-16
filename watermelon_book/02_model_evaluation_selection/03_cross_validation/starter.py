"""学生练习：通用交叉验证、折外预测和候选方案选择。"""

from collections.abc import Callable

import numpy as np


Fold = tuple[np.ndarray, np.ndarray]


def cross_validation_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """返回每个验证折的度量值。"""
    raise NotImplementedError("请完成 cross_validation_scores")


def out_of_fold_predictions(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
) -> np.ndarray:
    """按原始样本顺序返回每个样本恰好一次的折外预测。"""
    raise NotImplementedError("请完成 out_of_fold_predictions")


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    """返回折分数的均值和总体标准差。"""
    raise NotImplementedError("请完成 summarize_scores")


def select_best_candidate(
    scores_by_name: dict[str, np.ndarray], *, higher_is_better: bool
) -> tuple[str, dict[str, dict[str, float]]]:
    """按平均分选择候选方案，并返回全部摘要。"""
    raise NotImplementedError("请完成 select_best_candidate")
