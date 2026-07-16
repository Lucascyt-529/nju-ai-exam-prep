"""学生练习：带缺失值和样本权重的离散决策树。"""

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


def fit_missing_value_tree(
    X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None
) -> Model:
    raise NotImplementedError("请完成 fit_missing_value_tree")


def predict_proba_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_proba_missing_tree")


def predict_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_missing_tree")
