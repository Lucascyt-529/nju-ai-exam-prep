"""学生练习：带缺失值和样本权重的离散/连续决策树。"""

from typing import Any

import numpy as np


Model = dict[str, Any]


def weighted_entropy(y: np.ndarray, sample_weights: np.ndarray) -> float:
    raise NotImplementedError("请完成 weighted_entropy")


def branch_weight_plan(
    feature: np.ndarray, sample_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回取值、分支比例和(n,V)传播权重矩阵。"""
    raise NotImplementedError("请完成 branch_weight_plan")


def missing_information_gain(
    feature: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> float:
    raise NotImplementedError("请完成 missing_information_gain")


def continuous_missing_information_gain(
    feature: np.ndarray,
    y: np.ndarray,
    threshold: float,
    sample_weights: np.ndarray | None = None,
) -> float:
    raise NotImplementedError("请完成 continuous_missing_information_gain")


def best_continuous_missing_split(
    feature: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    raise NotImplementedError("请完成 best_continuous_missing_split")


def continuous_branch_weight_plan(
    feature: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 continuous_branch_weight_plan")


def fit_missing_value_tree(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
    *,
    feature_types: list[str] | None = None,
) -> Model:
    raise NotImplementedError("请完成 fit_missing_value_tree")


def predict_proba_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_proba_missing_tree")


def predict_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_missing_tree")
