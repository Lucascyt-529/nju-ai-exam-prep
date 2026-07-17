"""学生练习：单隐层网络的标准BP、累积BP训练与预测。"""

import numpy as np


def apply_gradients(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 apply_gradients")


def train_network(
    X: np.ndarray,
    y_column: np.ndarray,
    *,
    n_hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    raise NotImplementedError("请完成 train_network")


def make_epoch_sample_orders(
    n_samples: int,
    epochs: int,
    *,
    shuffle: bool = False,
    random_state: int = 0,
) -> tuple[np.ndarray, ...]:
    raise NotImplementedError("请完成 make_epoch_sample_orders")


def train_network_accumulated_bp(
    X: np.ndarray,
    y_column: np.ndarray,
    *,
    n_hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    raise NotImplementedError("请完成 train_network_accumulated_bp")


def train_network_standard_bp(
    X: np.ndarray,
    y_column: np.ndarray,
    *,
    n_hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int = 0,
    shuffle: bool = False,
    random_state: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    raise NotImplementedError("请完成 train_network_standard_bp")


def predict_probabilities(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_probabilities")


def predict_labels(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_labels")
