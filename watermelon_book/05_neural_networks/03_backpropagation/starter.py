"""学生练习：单隐层网络BP与数值梯度。"""

import numpy as np


def forward_pass(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 forward_pass")


def backward_pass(
    parameters: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    y_column: np.ndarray,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 backward_pass")


def finite_difference_gradients(
    X: np.ndarray,
    y_column: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    epsilon: float = 1e-5,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 finite_difference_gradients")


def maximum_relative_error(
    analytic: dict[str, np.ndarray], numerical: dict[str, np.ndarray]
) -> float:
    raise NotImplementedError("请完成 maximum_relative_error")
