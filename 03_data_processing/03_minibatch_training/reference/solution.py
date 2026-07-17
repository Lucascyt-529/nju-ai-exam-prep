"""参考实现：可复现mini-batch下标与线性回归训练数据流。"""

from numbers import Real

import numpy as np


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("random_state必须是整数")
    return int(value)


def batch_index_schedule(
    n_samples: int,
    batch_size: int,
    n_epochs: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    random_state: int = 0,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """用同一个RNG连续生成多个epoch，避免每轮重新播种。"""
    n = _positive_int(n_samples, "n_samples")
    size = _positive_int(batch_size, "batch_size")
    epochs = _positive_int(n_epochs, "n_epochs")
    if not isinstance(shuffle, (bool, np.bool_)) or not isinstance(drop_last, (bool, np.bool_)):
        raise ValueError("shuffle和drop_last必须是布尔值")
    rng = np.random.default_rng(_seed(random_state))
    schedule: list[tuple[np.ndarray, ...]] = []
    for _ in range(epochs):
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        batches = []
        for start in range(0, n, size):
            batch = indices[start:start + size]
            if len(batch) < size and drop_last:
                continue
            batches.append(batch.copy())
        if not batches:
            raise ValueError("drop_last=True会丢弃该epoch的全部样本")
        schedule.append(tuple(batches))
    return tuple(schedule)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> None:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X))):
        raise ValueError("X必须是非空有限数值二维数组")
    if (not isinstance(y, np.ndarray) or y.shape != (X.shape[0],)
            or not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y))):
        raise ValueError("y必须是与X样本数一致的有限数值一维数组")


def take_minibatch(X: np.ndarray, y: np.ndarray,
                   indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_xy(X, y)
    if (not isinstance(indices, np.ndarray) or indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(X))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("indices必须是范围内、非空且不重复的一维整数数组")
    return X[indices].copy(), y[indices].copy()


def linear_mse_loss_gradient(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[float, np.ndarray, float]:
    _validate_xy(X, y)
    if (not isinstance(weights, np.ndarray) or weights.shape != (X.shape[1],)
            or not np.issubdtype(weights.dtype, np.number) or not np.all(np.isfinite(weights))
            or isinstance(bias, (bool, np.bool_)) or not isinstance(bias, Real)
            or not np.isfinite(bias)):
        raise ValueError("weights或bias形状/数值无效")
    residual = X.astype(float) @ weights.astype(float) + float(bias) - y.astype(float)
    loss = 0.5 * float(np.mean(residual * residual))
    gradient_weights = X.astype(float).T @ residual / len(X)
    gradient_bias = float(np.mean(residual))
    return loss, gradient_weights, gradient_bias


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
    _validate_xy(X, y)
    if (isinstance(learning_rate, (bool, np.bool_)) or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate必须是正有限实数")
    schedule = batch_index_schedule(
        len(X), batch_size, n_epochs, shuffle=shuffle,
        drop_last=False, random_state=random_state,
    )
    weights = np.zeros(X.shape[1], dtype=float); bias = 0.0
    history = [linear_mse_loss_gradient(X, y, weights, bias)[0]]
    for epoch_batches in schedule:
        for indices in epoch_batches:
            X_batch, y_batch = take_minibatch(X, y, indices)
            _, gradient_weights, gradient_bias = linear_mse_loss_gradient(
                X_batch, y_batch, weights, bias
            )
            weights = weights - float(learning_rate) * gradient_weights
            bias -= float(learning_rate) * gradient_bias
        history.append(linear_mse_loss_gradient(X, y, weights, bias)[0])
    return {
        "weights": weights,
        "bias": float(bias),
        "loss_history": np.asarray(history),
        "batch_schedule": schedule,
    }
