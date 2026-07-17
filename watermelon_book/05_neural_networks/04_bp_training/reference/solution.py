"""参考实现：单隐层网络的标准BP、累积BP训练与预测。"""

import numpy as np


PARAMETER_KEYS = {"W1", "b1", "W2", "b2"}


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _validate_X(X: np.ndarray) -> None:
    if (
        not _is_finite_numeric_array(X)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
    ):
        raise ValueError("X必须是非空有限数值二维数组")


def _validate_X_y(X: np.ndarray, y_column: np.ndarray) -> None:
    _validate_X(X)
    if (
        not _is_finite_numeric_array(y_column)
        or y_column.shape != (X.shape[0], 1)
        or not np.all(np.isin(y_column, [0, 1]))
    ):
        raise ValueError("y_column必须是与X样本数一致的(n,1)二进制标签")


def _validate_parameters(parameters: dict[str, np.ndarray], n_features: int) -> None:
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("parameters必须恰好包含W1、b1、W2和b2")
    if not all(_is_finite_numeric_array(parameters[key]) for key in PARAMETER_KEYS):
        raise ValueError("所有参数必须是有限数值NumPy数组")
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )
    if W1.ndim != 2 or W1.shape[0] != n_features or W1.shape[1] == 0:
        raise ValueError("W1必须具有形状(d,h)，且h为正")
    n_hidden = W1.shape[1]
    if b1.shape != (n_hidden,) or W2.shape != (n_hidden, 1) or b2.shape != (1,):
        raise ValueError("参数形状必须是W1:(d,h)、b1:(h,)、W2:(h,1)、b2:(1,)")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    if not _is_finite_numeric_array(values) or values.size == 0:
        raise ValueError("values必须是非空有限数值NumPy数组")
    values_float = values.astype(float, copy=False)
    result = np.empty_like(values_float)
    nonnegative = values_float >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values_float[nonnegative]))
    exponent = np.exp(values_float[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def initialize_parameters(
    n_features: int, n_hidden: int, *, seed: int = 0
) -> dict[str, np.ndarray]:
    if not _is_integer(n_features) or n_features <= 0 or not _is_integer(n_hidden) or n_hidden <= 0:
        raise ValueError("n_features和n_hidden必须是正整数")
    if not _is_integer(seed):
        raise ValueError("seed必须是整数")
    generator = np.random.default_rng(int(seed))
    return {
        "W1": generator.normal(
            0.0, 1.0 / np.sqrt(n_features), size=(n_features, n_hidden)
        ),
        "b1": np.zeros(n_hidden),
        "W2": generator.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1)),
        "b2": np.zeros(1),
    }


def _forward_unchecked(
    X_float: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    z1 = X_float @ parameters["W1"] + parameters["b1"]
    a1 = stable_sigmoid(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    probabilities = stable_sigmoid(z2)
    return {
        "X": X_float,
        "a1": a1,
        "z2": z2,
        "probabilities": probabilities,
    }


def forward_pass(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    _validate_X(X)
    _validate_parameters(parameters, X.shape[1])
    return _forward_unchecked(X.astype(float, copy=True), parameters)


def _loss_from_logits(y_column: np.ndarray, logits: np.ndarray) -> float:
    losses = (
        np.maximum(logits, 0.0)
        - logits * y_column
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(np.mean(losses))


def _backward_unchecked(
    parameters: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    y_column: np.ndarray,
) -> dict[str, np.ndarray]:
    sample_count = cache["X"].shape[0]
    dz2 = (cache["probabilities"] - y_column) / sample_count
    dW2 = cache["a1"].T @ dz2
    db2 = np.sum(dz2, axis=0)
    da1 = dz2 @ parameters["W2"].T
    dz1 = da1 * cache["a1"] * (1.0 - cache["a1"])
    dW1 = cache["X"].T @ dz1
    db1 = np.sum(dz1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def _validate_gradients(
    gradients: dict[str, np.ndarray], parameters: dict[str, np.ndarray]
) -> None:
    if not isinstance(gradients, dict) or set(gradients) != PARAMETER_KEYS:
        raise ValueError("gradients必须恰好包含W1、b1、W2和b2")
    if any(
        not _is_finite_numeric_array(gradients[key])
        or gradients[key].shape != parameters[key].shape
        for key in PARAMETER_KEYS
    ):
        raise ValueError("每个梯度都必须是与对应参数同形状的有限数值数组")


def apply_gradients(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    if not isinstance(parameters, dict) or "W1" not in parameters or not isinstance(parameters["W1"], np.ndarray):
        raise ValueError("parameters格式无效")
    n_features = parameters["W1"].shape[0] if parameters["W1"].ndim == 2 else -1
    _validate_parameters(parameters, n_features)
    _validate_gradients(gradients, parameters)
    if not _is_finite_scalar(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate必须是正有限数值")
    return {
        key: parameters[key].astype(float, copy=True) - float(learning_rate) * gradients[key]
        for key in ("W1", "b1", "W2", "b2")
    }


def train_network(
    X: np.ndarray,
    y_column: np.ndarray,
    *,
    n_hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    _validate_X_y(X, y_column)
    if not _is_integer(n_hidden) or n_hidden <= 0:
        raise ValueError("n_hidden必须是正整数")
    if not _is_finite_scalar(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate必须是正有限数值")
    if not _is_integer(epochs) or epochs < 0:
        raise ValueError("epochs必须是非负整数")
    if not _is_integer(seed):
        raise ValueError("seed必须是整数")

    X_float = X.astype(float, copy=True)
    y_float = y_column.astype(float, copy=True)
    parameters = initialize_parameters(X.shape[1], int(n_hidden), seed=int(seed))
    cache = _forward_unchecked(X_float, parameters)
    history = [_loss_from_logits(y_float, cache["z2"])]
    for _ in range(int(epochs)):
        gradients = _backward_unchecked(parameters, cache, y_float)
        parameters = apply_gradients(parameters, gradients, float(learning_rate))
        cache = _forward_unchecked(X_float, parameters)
        history.append(_loss_from_logits(y_float, cache["z2"]))
    return parameters, history


def make_epoch_sample_orders(
    n_samples: int,
    epochs: int,
    *,
    shuffle: bool = False,
    random_state: int = 0,
) -> tuple[np.ndarray, ...]:
    """生成标准BP每一轮的样本访问顺序。

    同一个随机数生成器会跨轮持续使用，不能在每一轮重新用同一种子初始化，
    否则每轮都会得到完全相同的伪随机顺序。
    """
    if not _is_integer(n_samples) or n_samples <= 0:
        raise ValueError("n_samples必须是正整数")
    if not _is_integer(epochs) or epochs < 0:
        raise ValueError("epochs必须是非负整数")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError("shuffle必须是布尔值")
    if not _is_integer(random_state):
        raise ValueError("random_state必须是整数")

    generator = np.random.default_rng(int(random_state))
    orders: list[np.ndarray] = []
    for _ in range(int(epochs)):
        if bool(shuffle):
            order = generator.permutation(int(n_samples))
        else:
            order = np.arange(int(n_samples))
        orders.append(order)
    return tuple(orders)


def train_network_accumulated_bp(
    X: np.ndarray,
    y_column: np.ndarray,
    *,
    n_hidden: int,
    learning_rate: float,
    epochs: int,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """累积BP：每轮先求整个训练集的平均梯度，再更新一次参数。"""
    return train_network(
        X,
        y_column,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )


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
    """标准BP：每处理一个样本就立即计算梯度并更新一次参数。

    为便于与累积BP公平比较，历史仍按完整训练轮记录：第0项是初始的
    全训练集损失，之后每一项是完成一整轮逐样本更新后的全训练集损失。
    """
    _validate_X_y(X, y_column)
    if not _is_integer(n_hidden) or n_hidden <= 0:
        raise ValueError("n_hidden必须是正整数")
    if not _is_finite_scalar(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate必须是正有限数值")
    if not _is_integer(epochs) or epochs < 0:
        raise ValueError("epochs必须是非负整数")
    if not _is_integer(seed):
        raise ValueError("seed必须是整数")

    orders = make_epoch_sample_orders(
        X.shape[0],
        int(epochs),
        shuffle=shuffle,
        random_state=random_state,
    )
    X_float = X.astype(float, copy=True)
    y_float = y_column.astype(float, copy=True)
    parameters = initialize_parameters(X.shape[1], int(n_hidden), seed=int(seed))
    full_cache = _forward_unchecked(X_float, parameters)
    history = [_loss_from_logits(y_float, full_cache["z2"])]

    for order in orders:
        for sample_index in order:
            index = int(sample_index)
            X_one = X_float[index : index + 1]
            y_one = y_float[index : index + 1]
            sample_cache = _forward_unchecked(X_one, parameters)
            gradients = _backward_unchecked(parameters, sample_cache, y_one)
            parameters = apply_gradients(parameters, gradients, float(learning_rate))
        full_cache = _forward_unchecked(X_float, parameters)
        history.append(_loss_from_logits(y_float, full_cache["z2"]))
    return parameters, history


def predict_probabilities(
    X: np.ndarray, parameters: dict[str, np.ndarray]
) -> np.ndarray:
    return forward_pass(X, parameters)["probabilities"]


def predict_labels(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    if not _is_finite_scalar(threshold) or threshold < 0 or threshold > 1:
        raise ValueError("threshold必须是0到1之间的有限数值")
    return (predict_probabilities(X, parameters) >= float(threshold)).astype(int)
