"""学生练习：Elman网络单步状态和序列前向传播。"""

import numpy as np


def initialize_elman_parameters(
    n_inputs: int,
    n_hidden: int,
    n_outputs: int,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 initialize_elman_parameters")


def elman_step(
    x_t: np.ndarray,
    previous_state: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 elman_step")


def forward_sequence(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 forward_sequence")


def sequence_mean_squared_error(targets: np.ndarray, outputs: np.ndarray) -> float:
    raise NotImplementedError("请完成 sequence_mean_squared_error")
