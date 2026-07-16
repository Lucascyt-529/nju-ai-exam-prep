"""学生练习：二项、配对、McNemar、Friedman与Nemenyi比较。"""

import numpy as np


def binomial_lower_tail(successes: int, trials: int, probability: float) -> float:
    raise NotImplementedError("请完成 binomial_lower_tail")


def paired_t_statistic(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    raise NotImplementedError("请完成 paired_t_statistic")


def corrected_resampled_t_statistic(
    differences: np.ndarray, n_train: int, n_validation: int
) -> float:
    raise NotImplementedError("请完成 corrected_resampled_t_statistic")


def mcnemar_disagreements(
    y_true: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray
) -> tuple[int, int]:
    """返回(A错B对, A对B错)。"""
    raise NotImplementedError("请完成 mcnemar_disagreements")


def mcnemar_exact_p_value(b: int, c: int) -> float:
    raise NotImplementedError("请完成 mcnemar_exact_p_value")


def ranks_per_dataset(
    scores: np.ndarray, *, higher_is_better: bool
) -> np.ndarray:
    raise NotImplementedError("请完成 ranks_per_dataset")


def friedman_statistic(ranks: np.ndarray) -> tuple[np.ndarray, float]:
    """返回各算法平均秩与Friedman卡方统计量。"""
    raise NotImplementedError("请完成 friedman_statistic")


def nemenyi_critical_difference(
    n_algorithms: int, n_datasets: int, q_alpha: float
) -> float:
    raise NotImplementedError("请完成 nemenyi_critical_difference")


def nemenyi_significant_pairs(
    names: list[str], average_ranks: np.ndarray, critical_difference: float
) -> list[tuple[str, str]]:
    raise NotImplementedError("请完成 nemenyi_significant_pairs")
