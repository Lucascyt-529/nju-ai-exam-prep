"""学生练习：连续值与离散/连续混合决策树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]


def candidate_thresholds(feature: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 candidate_thresholds")


def continuous_split_score(
    feature: np.ndarray,
    y: np.ndarray,
    threshold: float,
    *,
    criterion: str = "information_gain",
) -> float:
    raise NotImplementedError("请完成 continuous_split_score")


def best_continuous_split(
    feature: np.ndarray, y: np.ndarray, *, criterion: str = "information_gain"
) -> tuple[float, float]:
    raise NotImplementedError("请完成 best_continuous_split")


def fit_mixed_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: list[str],
    *,
    criterion: str = "information_gain",
) -> Tree:
    raise NotImplementedError("请完成 fit_mixed_tree")


def predict_mixed_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_mixed_tree")
