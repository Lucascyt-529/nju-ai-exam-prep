"""参考实现：Elman网络的单步与序列前向传播。"""

import numpy as np


PARAMETER_KEYS = {"Wx", "Wh", "bh", "Wy", "by"}


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _is_positive_integer(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value > 0
    )


def _validate_sequence(X: np.ndarray) -> None:
    if (
        not _is_finite_numeric_array(X)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
    ):
        raise ValueError("X必须是非空有限数值二维序列(T,d)")


def _validate_parameters(parameters: dict[str, np.ndarray], n_inputs: int) -> None:
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("parameters必须恰好包含Wx、Wh、bh、Wy和by")
    if not all(_is_finite_numeric_array(parameters[key]) for key in PARAMETER_KEYS):
        raise ValueError("所有参数必须是有限数值NumPy数组")
    Wx, Wh, bh, Wy, by = (
        parameters["Wx"],
        parameters["Wh"],
        parameters["bh"],
        parameters["Wy"],
        parameters["by"],
    )
    if Wx.ndim != 2 or Wx.shape[0] != n_inputs or Wx.shape[1] == 0:
        raise ValueError("Wx必须具有形状(d,h)，且h为正")
    n_hidden = Wx.shape[1]
    if (
        Wh.shape != (n_hidden, n_hidden)
        or bh.shape != (n_hidden,)
        or Wy.ndim != 2
        or Wy.shape[0] != n_hidden
        or Wy.shape[1] == 0
        or by.shape != (Wy.shape[1],)
    ):
        raise ValueError("参数形状必须是Wx:(d,h)、Wh:(h,h)、bh:(h,)、Wy:(h,o)、by:(o,)")


def initialize_elman_parameters(
    n_inputs: int,
    n_hidden: int,
    n_outputs: int,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if not all(_is_positive_integer(value) for value in (n_inputs, n_hidden, n_outputs)):
        raise ValueError("n_inputs、n_hidden和n_outputs必须是正整数")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed必须是整数")
    generator = np.random.default_rng(int(seed))
    return {
        "Wx": generator.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_inputs, n_hidden)),
        "Wh": generator.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, n_hidden)),
        "bh": np.zeros(n_hidden),
        "Wy": generator.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, n_outputs)),
        "by": np.zeros(n_outputs),
    }


def elman_step(
    x_t: np.ndarray,
    previous_state: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not _is_finite_numeric_array(x_t) or x_t.ndim != 1 or x_t.size == 0:
        raise ValueError("x_t必须是非空一维有限数值数组")
    _validate_parameters(parameters, x_t.size)
    n_hidden = parameters["Wh"].shape[0]
    if not _is_finite_numeric_array(previous_state) or previous_state.shape != (n_hidden,):
        raise ValueError("previous_state必须具有形状(h,)")
    state = np.tanh(
        x_t.astype(float, copy=False) @ parameters["Wx"]
        + previous_state.astype(float, copy=False) @ parameters["Wh"]
        + parameters["bh"]
    )
    output = state @ parameters["Wy"] + parameters["by"]
    return state, output


def forward_sequence(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    _validate_sequence(X)
    _validate_parameters(parameters, X.shape[1])
    n_hidden = parameters["Wh"].shape[0]
    n_outputs = parameters["Wy"].shape[1]
    if initial_state is None:
        state = np.zeros(n_hidden)
    elif _is_finite_numeric_array(initial_state) and initial_state.shape == (n_hidden,):
        state = initial_state.astype(float, copy=True)
    else:
        raise ValueError("initial_state必须是None或形状(h,)的有限数组")

    states = np.empty((X.shape[0] + 1, n_hidden), dtype=float)
    outputs = np.empty((X.shape[0], n_outputs), dtype=float)
    states[0] = state
    for time_index, x_t in enumerate(X):
        state, output = elman_step(x_t, state, parameters)
        states[time_index + 1] = state
        outputs[time_index] = output
    return {"states": states, "outputs": outputs}


def sequence_mean_squared_error(targets: np.ndarray, outputs: np.ndarray) -> float:
    if (
        not _is_finite_numeric_array(targets)
        or not _is_finite_numeric_array(outputs)
        or targets.ndim != 2
        or targets.shape != outputs.shape
        or targets.shape[0] == 0
        or targets.shape[1] == 0
    ):
        raise ValueError("targets和outputs必须是同形状非空二维有限数组")
    return float(np.mean((outputs - targets) ** 2))
