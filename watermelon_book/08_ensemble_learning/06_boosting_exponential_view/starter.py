"""学生练习：AdaBoost加性模型、指数风险与样本分布。"""

import numpy as np


def exponential_risk(y: np.ndarray, scores: np.ndarray, *, distribution: np.ndarray | None = None) -> float:
    raise NotImplementedError("请完成 exponential_risk")


def bayes_optimal_additive_score(p_positive: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 bayes_optimal_additive_score")


def round_exponential_loss(error: float, alpha: float) -> float:
    raise NotImplementedError("请完成 round_exponential_loss")


def optimal_alpha(error: float) -> float:
    raise NotImplementedError("请完成 optimal_alpha")


def update_distribution(distribution: np.ndarray, y: np.ndarray, prediction: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 update_distribution")


def distribution_from_scores(base_distribution: np.ndarray, y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 distribution_from_scores")


def trace_additive_rounds(y: np.ndarray, predictions: np.ndarray, alphas: np.ndarray, *, base_distribution: np.ndarray | None = None) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 trace_additive_rounds")


def training_error_and_exponential_bound(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError("请完成 training_error_and_exponential_bound")


def weighted_resample_indices(distribution: np.ndarray, *, n_samples: int | None = None, seed: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 weighted_resample_indices")
