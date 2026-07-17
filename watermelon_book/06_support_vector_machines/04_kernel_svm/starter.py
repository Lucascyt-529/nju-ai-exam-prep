"""学生练习：核矩阵、核化SMO与非线性预测。"""

import numpy as np


def kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    kernel: str = "linear",
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> np.ndarray:
    raise NotImplementedError("请完成 kernel_matrix")


def fit_kernel_svm_smo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    kernel: str = "rbf",
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    tolerance: float = 1e-3,
    max_passes: int = 10,
    max_iterations: int = 1000,
) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_kernel_svm_smo")


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function")


def predict_labels(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_labels")
