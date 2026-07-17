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


def _validate_targets(targets: np.ndarray, length: int, n_outputs: int) -> None:
    if not _is_finite_numeric_array(targets) or targets.shape != (length, n_outputs):
        raise ValueError("targets必须具有形状(T,o)且包含有限数值")


def elman_bptt(X, targets, parameters, *, initial_state=None):
    """对单条完整序列计算MSE及五组参数的BPTT梯度。"""
    forward = forward_sequence(X, parameters, initial_state=initial_state)
    outputs, states = forward["outputs"], forward["states"]
    _validate_targets(targets, X.shape[0], outputs.shape[1])
    gradients = {key: np.zeros_like(value, dtype=float) for key, value in parameters.items()}
    output_gradient = 2.0 * (outputs - targets) / outputs.size
    future_state_gradient = np.zeros(states.shape[1])
    for time_index in range(X.shape[0] - 1, -1, -1):
        gradients["Wy"] += np.outer(states[time_index + 1], output_gradient[time_index])
        gradients["by"] += output_gradient[time_index]
        state_gradient = output_gradient[time_index] @ parameters["Wy"].T + future_state_gradient
        preactivation_gradient = state_gradient * (1.0 - states[time_index + 1] ** 2)
        gradients["Wx"] += np.outer(X[time_index], preactivation_gradient)
        gradients["Wh"] += np.outer(states[time_index], preactivation_gradient)
        gradients["bh"] += preactivation_gradient
        future_state_gradient = preactivation_gradient @ parameters["Wh"].T
    return {"loss": sequence_mean_squared_error(targets, outputs), "gradients": gradients, "initial_state_gradient": future_state_gradient, "states": states, "outputs": outputs}


def gradient_check_elman(X, targets, parameters, *, initial_state=None, epsilon=1e-6):
    if not isinstance(epsilon, (int, float, np.integer, np.floating)) or isinstance(epsilon, (bool, np.bool_)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon必须是有限正数")
    analytic = elman_bptt(X, targets, parameters, initial_state=initial_state)["gradients"]
    errors = {}
    for key in PARAMETER_KEYS:
        numeric = np.zeros_like(parameters[key], dtype=float)
        for index in np.ndindex(parameters[key].shape):
            plus = {name: value.astype(float).copy() for name, value in parameters.items()}
            minus = {name: value.astype(float).copy() for name, value in parameters.items()}
            plus[key][index] += epsilon; minus[key][index] -= epsilon
            plus_output = forward_sequence(X, plus, initial_state=initial_state)["outputs"]
            minus_output = forward_sequence(X, minus, initial_state=initial_state)["outputs"]
            numeric[index] = (sequence_mean_squared_error(targets, plus_output) - sequence_mean_squared_error(targets, minus_output)) / (2.0 * epsilon)
        denominator = np.maximum(1.0, np.abs(numeric) + np.abs(analytic[key]))
        errors[key] = float(np.max(np.abs(numeric - analytic[key]) / denominator))
    return errors


def train_elman_sequence(X, targets, n_hidden, *, learning_rate=0.05, epochs=500, seed=0, clip_norm=5.0):
    _validate_sequence(X)
    if not _is_finite_numeric_array(targets) or targets.ndim != 2 or targets.shape[0] != X.shape[0] or targets.shape[1] == 0:
        raise ValueError("targets必须是与X等长的非空有限二维数组")
    if not _is_positive_integer(n_hidden) or not _is_positive_integer(epochs): raise ValueError("n_hidden和epochs必须是正整数")
    if not isinstance(learning_rate, (int, float, np.integer, np.floating)) or isinstance(learning_rate, (bool, np.bool_)) or not np.isfinite(learning_rate) or learning_rate <= 0: raise ValueError("learning_rate必须是有限正数")
    if clip_norm is not None and (not isinstance(clip_norm, (int, float, np.integer, np.floating)) or isinstance(clip_norm, (bool, np.bool_)) or not np.isfinite(clip_norm) or clip_norm <= 0): raise ValueError("clip_norm必须是None或有限正数")
    parameters = initialize_elman_parameters(X.shape[1], int(n_hidden), targets.shape[1], seed=seed)
    history = [sequence_mean_squared_error(targets, forward_sequence(X, parameters)["outputs"])]
    gradient_norms = []
    for _ in range(int(epochs)):
        gradients = elman_bptt(X, targets, parameters)["gradients"]
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients.values()))); gradient_norms.append(norm)
        scale = 1.0 if clip_norm is None or norm <= clip_norm else float(clip_norm) / norm
        parameters = {key: parameters[key] - float(learning_rate) * scale * gradients[key] for key in PARAMETER_KEYS}
        history.append(sequence_mean_squared_error(targets, forward_sequence(X, parameters)["outputs"]))
    return {"parameters": parameters, "loss_history": np.asarray(history), "gradient_norms": np.asarray(gradient_norms), "clip_norm": clip_norm}
