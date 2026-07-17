"""学生练习：单隐层网络前向传播和稳定损失。"""

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 stable_sigmoid")


def initialize_parameters(
    n_features: int, n_hidden: int, *, seed: int = 0
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 initialize_parameters")


def as_column_labels(y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 as_column_labels")


def forward_pass(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 forward_pass")


def binary_cross_entropy_from_logits(y_column: np.ndarray, logits: np.ndarray) -> float:
    raise NotImplementedError("请完成 binary_cross_entropy_from_logits")
