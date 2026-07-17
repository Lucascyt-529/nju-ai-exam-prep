"""参考实现：一维网格SOM竞争学习。"""

import numpy as np


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _validate_matrix(value: np.ndarray, name: str) -> None:
    if (
        not _is_finite_numeric_array(value)
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] == 0
    ):
        raise ValueError(f"{name}必须是非空有限数值二维数组")


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _positive_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _validate_sample(sample: np.ndarray, n_features: int) -> None:
    if (
        not _is_finite_numeric_array(sample)
        or sample.shape != (n_features,)
    ):
        raise ValueError("sample必须是与原型特征数一致的一维有限数组")


def best_matching_unit(sample: np.ndarray, prototypes: np.ndarray) -> int:
    _validate_matrix(prototypes, "prototypes")
    _validate_sample(sample, prototypes.shape[1])
    difference = prototypes.astype(float, copy=False) - sample.astype(float, copy=False)
    squared_distances = np.sum(difference**2, axis=1)
    return int(np.argmin(squared_distances))


def gaussian_neighborhood(winner: int, n_neurons: int, radius: float) -> np.ndarray:
    if not _is_integer(n_neurons) or n_neurons <= 0:
        raise ValueError("n_neurons必须是正整数")
    if not _is_integer(winner) or winner < 0 or winner >= n_neurons:
        raise ValueError("winner必须是有效神经元下标")
    radius_value = _positive_scalar(radius, "radius")
    positions = np.arange(n_neurons, dtype=float)
    return np.exp(-((positions - int(winner)) ** 2) / (2.0 * radius_value**2))


def update_prototypes(
    prototypes: np.ndarray,
    sample: np.ndarray,
    winner: int,
    *,
    learning_rate: float,
    radius: float,
) -> np.ndarray:
    _validate_matrix(prototypes, "prototypes")
    _validate_sample(sample, prototypes.shape[1])
    learning_rate_value = _positive_scalar(learning_rate, "learning_rate")
    if learning_rate_value > 1.0:
        raise ValueError("learning_rate必须不大于1")
    neighborhood = gaussian_neighborhood(winner, prototypes.shape[0], radius)
    prototypes_float = prototypes.astype(float, copy=True)
    return prototypes_float + learning_rate_value * neighborhood[:, None] * (
        sample.astype(float, copy=False)[None, :] - prototypes_float
    )


def exponential_decay(
    initial: float, final: float, step: int, total_steps: int
) -> float:
    initial_value = _positive_scalar(initial, "initial")
    final_value = _positive_scalar(final, "final")
    if final_value > initial_value:
        raise ValueError("final不能大于initial")
    if not _is_integer(total_steps) or total_steps <= 0:
        raise ValueError("total_steps必须是正整数")
    if not _is_integer(step) or step < 0 or step >= total_steps:
        raise ValueError("step必须位于[0,total_steps)范围内")
    if total_steps == 1:
        return initial_value
    fraction = int(step) / (int(total_steps) - 1)
    return initial_value * (final_value / initial_value) ** fraction


def initialize_prototypes(
    X: np.ndarray, n_neurons: int, *, seed: int = 0
) -> np.ndarray:
    _validate_matrix(X, "X")
    if not _is_integer(n_neurons) or n_neurons <= 0:
        raise ValueError("n_neurons必须是正整数")
    if not _is_integer(seed):
        raise ValueError("seed必须是整数")
    generator = np.random.default_rng(int(seed))
    lower = X.min(axis=0)
    upper = X.max(axis=0)
    return generator.uniform(lower, upper, size=(int(n_neurons), X.shape[1]))


def quantization_error(X: np.ndarray, prototypes: np.ndarray) -> float:
    _validate_matrix(X, "X")
    _validate_matrix(prototypes, "prototypes")
    if X.shape[1] != prototypes.shape[1]:
        raise ValueError("X和prototypes特征数必须一致")
    difference = X.astype(float, copy=False)[:, None, :] - prototypes.astype(float, copy=False)[None, :, :]
    distances = np.sqrt(np.sum(difference**2, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def train_som(
    X: np.ndarray,
    *,
    n_neurons: int,
    epochs: int,
    initial_learning_rate: float = 0.5,
    final_learning_rate: float = 0.05,
    initial_radius: float | None = None,
    final_radius: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    _validate_matrix(X, "X")
    if not _is_integer(n_neurons) or n_neurons <= 0:
        raise ValueError("n_neurons必须是正整数")
    if not _is_integer(epochs) or epochs < 0:
        raise ValueError("epochs必须是非负整数")
    if not _is_integer(seed):
        raise ValueError("seed必须是整数")
    initial_learning_rate_value = _positive_scalar(
        initial_learning_rate, "initial_learning_rate"
    )
    final_learning_rate_value = _positive_scalar(final_learning_rate, "final_learning_rate")
    if initial_learning_rate_value > 1 or final_learning_rate_value > 1:
        raise ValueError("学习率必须不大于1")
    if final_learning_rate_value > initial_learning_rate_value:
        raise ValueError("final_learning_rate不能大于initial_learning_rate")
    radius_start = (
        max(1.0, n_neurons / 2.0)
        if initial_radius is None
        else _positive_scalar(initial_radius, "initial_radius")
    )
    radius_end = _positive_scalar(final_radius, "final_radius")
    if radius_end > radius_start:
        raise ValueError("final_radius不能大于initial_radius")

    X_float = X.astype(float, copy=True)
    prototypes = initialize_prototypes(X_float, int(n_neurons), seed=int(seed))
    history = [quantization_error(X_float, prototypes)]
    total_updates = int(epochs) * X.shape[0]
    update_index = 0
    for _ in range(int(epochs)):
        for sample in X_float:
            learning_rate = exponential_decay(
                initial_learning_rate_value,
                final_learning_rate_value,
                update_index,
                total_updates,
            )
            radius = exponential_decay(
                radius_start, radius_end, update_index, total_updates
            )
            winner = best_matching_unit(sample, prototypes)
            prototypes = update_prototypes(
                prototypes,
                sample,
                winner,
                learning_rate=learning_rate,
                radius=radius,
            )
            update_index += 1
        history.append(quantization_error(X_float, prototypes))
    return prototypes, history


def map_samples(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    _validate_matrix(X, "X")
    _validate_matrix(prototypes, "prototypes")
    if X.shape[1] != prototypes.shape[1]:
        raise ValueError("X和prototypes特征数必须一致")
    return np.array([best_matching_unit(sample, prototypes) for sample in X], dtype=int)
