"""学生练习：RBF隐藏层、线性输出拟合与预测。"""

import numpy as np


def squared_euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 squared_euclidean_distances")


def rbf_design_matrix(
    X: np.ndarray,
    centers: np.ndarray,
    width: float,
    *,
    include_bias: bool = True,
) -> np.ndarray:
    raise NotImplementedError("请完成 rbf_design_matrix")


def fit_rbf_output(
    X: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    width: float,
    *,
    regularization: float = 0.0,
) -> dict[str, np.ndarray | float]:
    raise NotImplementedError("请完成 fit_rbf_output")


def predict_rbf(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_rbf")
