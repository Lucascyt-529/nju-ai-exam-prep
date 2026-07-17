"""学生练习：集成结合策略与多样性度量。"""

import numpy as np


def weighted_average(predictions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError("请完成 weighted_average")


def hard_vote(predictions: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 hard_vote")


def pairwise_contingency(y: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, int]:
    raise NotImplementedError("请完成 pairwise_contingency")


def regression_error_ambiguity(y: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 regression_error_ambiguity")
