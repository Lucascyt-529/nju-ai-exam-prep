"""参考实现：平均/投票、成对多样性与误差-分歧分解。"""

import numpy as np


def _finite(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value))


def _prediction_matrix(predictions: np.ndarray) -> None:
    if not _finite(predictions) or predictions.ndim != 2 or 0 in predictions.shape:
        raise ValueError("predictions必须是非空有限数值二维数组(B,n)")


def _normalized_weights(weights: np.ndarray | None, B: int) -> np.ndarray:
    if weights is None:
        return np.full(B, 1.0/B)
    if not _finite(weights) or weights.shape != (B,) or np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights必须是形状(B,)且和为1的非负有限数组")
    return weights.astype(float, copy=False)


def weighted_average(predictions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    _prediction_matrix(predictions)
    return _normalized_weights(weights, predictions.shape[0]) @ predictions.astype(float)


def hard_vote(predictions: np.ndarray) -> np.ndarray:
    _prediction_matrix(predictions)
    output = []
    for column in predictions.T:
        labels, counts = np.unique(column, return_counts=True)
        output.append(labels[np.argmax(counts)])
    return np.asarray(output, dtype=predictions.dtype)


def soft_vote(probabilities: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not _finite(probabilities) or probabilities.ndim != 3 or 0 in probabilities.shape or np.any(probabilities < 0) or not np.allclose(np.sum(probabilities, axis=2), 1.0):
        raise ValueError("probabilities必须是形状(B,n,K)且每个分布和为1")
    w = _normalized_weights(weights, probabilities.shape[0])
    combined = np.tensordot(w, probabilities, axes=(0, 0))
    return combined, np.argmax(combined, axis=1)


def pairwise_contingency(y: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, int]:
    if not _finite(y) or y.ndim != 1 or y.size == 0 or not _finite(prediction_a) or prediction_a.shape != y.shape or not _finite(prediction_b) or prediction_b.shape != y.shape:
        raise ValueError("y和两个预测必须是同形状非空有限一维数组")
    a, b = prediction_a == y, prediction_b == y
    return {"n11": int(np.count_nonzero(a & b)), "n00": int(np.count_nonzero(~a & ~b)),
            "n10": int(np.count_nonzero(a & ~b)), "n01": int(np.count_nonzero(~a & b))}


def diversity_metrics(counts: dict[str, int]) -> dict[str, float]:
    if not isinstance(counts, dict) or set(counts) != {"n11", "n00", "n10", "n01"} or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 0 for v in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("counts必须包含四个非负整数且总数大于0")
    n11, n00, n10, n01 = (counts[key] for key in ("n11", "n00", "n10", "n01")); n = n11+n00+n10+n01
    denominator = n11*n00 + n10*n01
    q = (n11*n00 - n10*n01)/denominator if denominator else float("nan")
    corr_den = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    corr = (n11*n00-n10*n01)/corr_den if corr_den else float("nan")
    return {"q": float(q), "correlation": float(corr), "disagreement": (n10+n01)/n, "double_fault": n00/n}


def regression_error_ambiguity(y: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    if not _finite(y) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空有限一维数组")
    _prediction_matrix(predictions)
    if predictions.shape[1] != y.size:
        raise ValueError("predictions样本数必须与y一致")
    ensemble = np.mean(predictions, axis=0)
    individual_error = np.mean((predictions-y[None, :])**2, axis=0)
    ambiguity = np.mean((predictions-ensemble[None, :])**2, axis=0)
    ensemble_error = (ensemble-y)**2
    return {"ensemble_prediction": ensemble, "mean_individual_error": individual_error,
            "ambiguity": ambiguity, "ensemble_error": ensemble_error}
