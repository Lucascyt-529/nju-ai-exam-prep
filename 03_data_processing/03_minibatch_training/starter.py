"""学生练习：mini-batch下标、同步取样与训练循环。"""

import numpy as np


def batch_index_schedule(
    n_samples: int,
    batch_size: int,
    n_epochs: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    random_state: int = 0,
) -> tuple[tuple[np.ndarray, ...], ...]:
    raise NotImplementedError("请完成 batch_index_schedule")


def take_minibatch(X: np.ndarray, y: np.ndarray,
                   indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 take_minibatch")


def linear_mse_loss_gradient(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[float, np.ndarray, float]:
    raise NotImplementedError("请完成 linear_mse_loss_gradient")


def fit_minibatch_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    n_epochs: int = 100,
    learning_rate: float = 0.05,
    shuffle: bool = True,
    random_state: int = 0,
) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_minibatch_linear_regression")
