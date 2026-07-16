"""学生练习：稳定的NumPy对数几率回归。"""

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 stable_sigmoid")


def binary_cross_entropy_from_logits(
    y: np.ndarray, logits: np.ndarray
) -> float:
    raise NotImplementedError("请完成 binary_cross_entropy_from_logits")


def predict_proba(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_proba")


def logistic_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
) -> float:
    raise NotImplementedError("请完成 logistic_loss")


def logistic_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 logistic_gradients")


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    l2: float = 0.0,
    initial_weights: np.ndarray | None = None,
    initial_bias: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """返回权重、截距和包含初始损失的历史。"""
    raise NotImplementedError("请完成 fit_gradient_descent")


def predict_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    raise NotImplementedError("请完成 predict_labels")
