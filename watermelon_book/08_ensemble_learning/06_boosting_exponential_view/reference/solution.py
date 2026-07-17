"""参考实现：AdaBoost的加性模型、指数风险、分布更新与重采样视角。"""

import numpy as np


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or not np.issubdtype(value.dtype, np.number)
        or value.ndim != 1
        or value.size == 0
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name}必须是非空有限数值一维数组")
    return value.astype(float, copy=False)


def _labels(value: np.ndarray, name: str) -> np.ndarray:
    result = _vector(value, name)
    if not np.all(np.isin(result, [-1, 1])):
        raise ValueError(f"{name}只能包含-1和+1")
    return result


def _distribution(value: np.ndarray, n: int) -> np.ndarray:
    result = _vector(value, "distribution")
    if result.shape != (n,) or np.any(result < 0) or not np.isclose(result.sum(), 1.0):
        raise ValueError("distribution必须是形状(n,)且和为1的非负数组")
    return result


def exponential_risk(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    distribution: np.ndarray | None = None,
) -> float:
    labels, values = _labels(y, "y"), _vector(scores, "scores")
    if labels.shape != values.shape:
        raise ValueError("y和scores形状必须一致")
    weights = (
        np.full(labels.size, 1.0 / labels.size)
        if distribution is None
        else _distribution(distribution, labels.size)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.exp(-labels * values)
    risk = float(weights @ terms)
    if not np.isfinite(risk):
        raise ValueError("指数风险溢出")
    return risk


def bayes_optimal_additive_score(p_positive: np.ndarray) -> np.ndarray:
    probabilities = _vector(p_positive, "p_positive")
    if np.any(probabilities <= 0) or np.any(probabilities >= 1):
        raise ValueError("p_positive必须严格位于(0,1)")
    return 0.5 * (np.log(probabilities) - np.log1p(-probabilities))


def round_exponential_loss(error: float, alpha: float) -> float:
    if (
        not isinstance(error, (int, float, np.integer, np.floating))
        or isinstance(error, (bool, np.bool_))
        or not np.isfinite(error)
        or not 0 <= error <= 1
    ):
        raise ValueError("error必须位于[0,1]")
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
    ):
        raise ValueError("alpha必须是有限数值")
    value = np.exp(-float(alpha)) * (1.0 - float(error)) + np.exp(float(alpha)) * float(error)
    if not np.isfinite(value):
        raise ValueError("单轮指数损失溢出")
    return float(value)


def optimal_alpha(error: float) -> float:
    if (
        not isinstance(error, (int, float, np.integer, np.floating))
        or isinstance(error, (bool, np.bool_))
        or not np.isfinite(error)
        or not 0 < error < 1
    ):
        raise ValueError("推导中的error必须严格位于(0,1)")
    return float(0.5 * np.log((1.0 - float(error)) / float(error)))


def update_distribution(
    distribution: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    labels, predicted = _labels(y, "y"), _labels(prediction, "prediction")
    if labels.shape != predicted.shape:
        raise ValueError("y和prediction形状必须一致")
    weights = _distribution(distribution, labels.size)
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
    ):
        raise ValueError("alpha必须是有限数值")
    raw = weights * np.exp(-float(alpha) * labels * predicted)
    normalizer = float(raw.sum())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("归一化因子必须是正有限数值")
    return raw / normalizer, normalizer


def distribution_from_scores(
    base_distribution: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    labels, values = _labels(y, "y"), _vector(scores, "scores")
    if labels.shape != values.shape:
        raise ValueError("y和scores形状必须一致")
    base = _distribution(base_distribution, labels.size)
    raw = base * np.exp(-labels * values)
    if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
        raise ValueError("分布重建出现非有限值")
    return raw / raw.sum()


def trace_additive_rounds(
    y: np.ndarray,
    predictions: np.ndarray,
    alphas: np.ndarray,
    *,
    base_distribution: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    labels = _labels(y, "y")
    if (
        not isinstance(predictions, np.ndarray)
        or not np.issubdtype(predictions.dtype, np.number)
        or predictions.ndim != 2
        or predictions.shape[1] != labels.size
        or not np.all(np.isin(predictions, [-1, 1]))
    ):
        raise ValueError("predictions必须是形状(T,n)的-1/+1数组")
    alpha_values = _vector(alphas, "alphas")
    if alpha_values.shape != (predictions.shape[0],):
        raise ValueError("alphas长度必须等于轮数T")
    base = (
        np.full(labels.size, 1.0 / labels.size)
        if base_distribution is None
        else _distribution(base_distribution, labels.size).copy()
    )
    scores = np.zeros(labels.size)
    distributions = [base.copy()]
    score_history = [scores.copy()]
    risks = [exponential_risk(labels, scores, distribution=base)]
    normalizers = []
    current = base.copy()
    for prediction, alpha in zip(predictions, alpha_values, strict=True):
        current, normalizer = update_distribution(current, labels, prediction, float(alpha))
        scores = scores + float(alpha) * prediction
        distributions.append(current.copy())
        score_history.append(scores.copy())
        risks.append(exponential_risk(labels, scores, distribution=base))
        normalizers.append(normalizer)
    return {
        "scores": np.vstack(score_history),
        "distributions": np.vstack(distributions),
        "risks": np.asarray(risks),
        "normalizers": np.asarray(normalizers),
        "normalizer_products": np.concatenate(([1.0], np.cumprod(normalizers))),
    }


def training_error_and_exponential_bound(
    y: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    labels, values = _labels(y, "y"), _vector(scores, "scores")
    if labels.shape != values.shape:
        raise ValueError("y和scores形状必须一致")
    error = float(np.mean(labels * values <= 0.0))
    return error, exponential_risk(labels, values)


def weighted_resample_indices(
    distribution: np.ndarray,
    *,
    n_samples: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    weights = _vector(distribution, "distribution")
    weights = _distribution(weights, weights.size)
    count = weights.size if n_samples is None else n_samples
    if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count <= 0:
        raise ValueError("n_samples必须是正整数")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed必须是整数")
    rng = np.random.default_rng(int(seed))
    return rng.choice(weights.size, size=int(count), replace=True, p=weights)
