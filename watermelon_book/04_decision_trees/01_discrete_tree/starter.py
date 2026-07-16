"""学生练习：离散特征决策树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]


def entropy(y: np.ndarray) -> float:
    raise NotImplementedError("请完成 entropy")


def gini(y: np.ndarray) -> float:
    raise NotImplementedError("请完成 gini")


def majority_label(y: np.ndarray):
    raise NotImplementedError("请完成 majority_label")


def information_gain(feature: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("请完成 information_gain")


def gain_ratio(feature: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("请完成 gain_ratio")


def gini_index(feature: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("请完成 gini_index")


def choose_best_feature(
    X: np.ndarray,
    y: np.ndarray,
    available_features: np.ndarray,
    *,
    criterion: str = "information_gain",
) -> tuple[int, float]:
    raise NotImplementedError("请完成 choose_best_feature")


def fit_discrete_tree(
    X: np.ndarray, y: np.ndarray, *, criterion: str = "information_gain"
) -> Tree:
    raise NotImplementedError("请完成 fit_discrete_tree")


def predict_discrete_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_discrete_tree")
