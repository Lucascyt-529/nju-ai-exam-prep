"""学生练习：线性SVM对偶目标与简化SMO。"""

import numpy as np


def linear_kernel_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 linear_kernel_matrix")


def dual_objective(alphas: np.ndarray, y: np.ndarray, gram: np.ndarray) -> float:
    raise NotImplementedError("请完成 dual_objective")


def fit_linear_svm_smo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    tolerance: float = 1e-3,
    max_passes: int = 10,
    max_iterations: int = 1000,
) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_linear_svm_smo")


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function")


def linear_weights(model: dict[str, object]) -> np.ndarray:
    raise NotImplementedError("请完成 linear_weights")


def kkt_residuals(model: dict[str, object], *, alpha_tolerance: float = 1e-7) -> np.ndarray:
    raise NotImplementedError("请完成 kkt_residuals")
