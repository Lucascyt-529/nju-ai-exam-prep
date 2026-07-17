"""参考实现：类别、Bernoulli、类别分布与高斯MLE。"""

import numpy as np


def _finite_numeric(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value))


def _vector(value: np.ndarray, name: str) -> None:
    if not _finite_numeric(value) or value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name}必须是非空有限数值一维数组")


def class_prior_mle(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _vector(y, "y")
    classes, counts = np.unique(y, return_counts=True)
    return classes, counts.astype(float) / y.size


def bernoulli_mle(samples: np.ndarray) -> float:
    _vector(samples, "samples")
    if not np.all(np.isin(samples, [0, 1])):
        raise ValueError("Bernoulli样本只能包含0和1")
    return float(np.mean(samples))


def _nonnegative(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name}必须是非负有限数值")
    return float(value)


def categorical_probabilities(samples: np.ndarray, categories: np.ndarray, *, alpha: float = 0.0) -> np.ndarray:
    _vector(samples, "samples")
    _vector(categories, "categories")
    alpha_value = _nonnegative(alpha, "alpha")
    if np.unique(categories).size != categories.size:
        raise ValueError("categories不能包含重复值")
    if not np.all(np.isin(samples, categories)):
        raise ValueError("samples包含categories之外的取值")
    counts = np.array([np.count_nonzero(samples == category) for category in categories], dtype=float)
    return (counts + alpha_value) / (samples.size + alpha_value * categories.size)


def gaussian_mle(samples: np.ndarray) -> tuple[float, float]:
    _vector(samples, "samples")
    values = samples.astype(float, copy=False)
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    return mean, variance


def multivariate_gaussian_mle(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _finite_numeric(X) or X.ndim != 2 or 0 in X.shape:
        raise ValueError("X必须是非空有限数值二维数组")
    values = X.astype(float, copy=False)
    mean = np.mean(values, axis=0)
    centered = values - mean
    covariance = centered.T @ centered / X.shape[0]
    return mean, covariance


def bernoulli_log_likelihood(samples: np.ndarray, probability: float) -> float:
    _vector(samples, "samples")
    if not np.all(np.isin(samples, [0, 1])):
        raise ValueError("Bernoulli样本只能包含0和1")
    if not isinstance(probability, (int, float, np.integer, np.floating)) or isinstance(probability, (bool, np.bool_)) or not np.isfinite(probability) or probability < 0 or probability > 1:
        raise ValueError("probability必须位于[0,1]")
    ones = int(np.sum(samples))
    zeros = samples.size - ones
    if (probability == 0 and ones > 0) or (probability == 1 and zeros > 0):
        return float("-inf")
    result = 0.0
    if ones:
        result += ones * np.log(probability)
    if zeros:
        result += zeros * np.log1p(-probability)
    return float(result)
