"""学生练习：多变量线性划分决策树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]


def projection_values(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 projection_values")


def best_projection_threshold(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    min_samples_leaf: int = 1,
) -> tuple[float, float]:
    """返回threshold和加权gini。"""
    raise NotImplementedError("请完成 best_projection_threshold")


def coordinate_search_split(
    X: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    *,
    min_samples_leaf: int = 1,
    initial_step: float = 1.0,
    max_rounds: int = 8,
) -> tuple[np.ndarray, float, float]:
    raise NotImplementedError("请完成 coordinate_search_split")


def find_linear_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    min_samples_leaf: int = 1,
    max_search_rounds: int = 8,
) -> tuple[np.ndarray, float, float]:
    raise NotImplementedError("请完成 find_linear_split")


def fit_multivariate_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    max_search_rounds: int = 8,
) -> Tree:
    raise NotImplementedError("请完成 fit_multivariate_tree")


def predict_multivariate_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_multivariate_tree")
