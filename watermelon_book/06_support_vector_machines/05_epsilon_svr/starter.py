"""学生练习：epsilon不敏感损失与核SVR。"""

import numpy as np


def epsilon_insensitive_loss(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float
) -> np.ndarray:
    raise NotImplementedError("请完成 epsilon_insensitive_loss")


def fit_epsilon_svr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    epsilon: float = 0.1,
    kernel: str = "linear",
    gamma: float | None = None,
    tolerance: float = 1e-7,
    max_passes: int = 100,
) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_epsilon_svr")


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function")


def kkt_residuals(model: dict[str, object]) -> np.ndarray:
    raise NotImplementedError("请完成 kkt_residuals")
