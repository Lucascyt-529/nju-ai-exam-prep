"""学生练习：SVM超平面得分、函数间隔与几何间隔。"""

import numpy as np


def decision_scores(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    raise NotImplementedError("请完成 decision_scores")


def functional_margins(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    raise NotImplementedError("请完成 functional_margins")


def geometric_margins(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    raise NotImplementedError("请完成 geometric_margins")


def canonical_rescale(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 canonical_rescale")


def minimum_margin_indices(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    raise NotImplementedError("请完成 minimum_margin_indices")
