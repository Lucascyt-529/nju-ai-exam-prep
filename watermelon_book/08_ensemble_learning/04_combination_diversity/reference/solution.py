"""参考实现：平均/投票、成对多样性与误差-分歧分解。"""

from numbers import Real
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


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def _random_state(random_state: int) -> int:
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state必须是整数")
    return int(random_state)


def bootstrap_index_matrix(n_samples: int, n_learners: int, *,
                           random_state: int = 0) -> np.ndarray:
    """数据样本扰动：每个学习器得到一份有放回样本下标。"""
    n = _positive_integer(n_samples, "n_samples"); learners = _positive_integer(n_learners, "n_learners")
    rng = np.random.default_rng(_random_state(random_state))
    return rng.integers(0, n, size=(learners, n))


def random_feature_subspaces(n_features: int, n_learners: int, subspace_size: int, *,
                             random_state: int = 0) -> np.ndarray:
    """输入属性扰动：为每个学习器无放回选择并排序特征下标。"""
    features = _positive_integer(n_features, "n_features")
    learners = _positive_integer(n_learners, "n_learners")
    size = _positive_integer(subspace_size, "subspace_size")
    if size > features:
        raise ValueError("subspace_size不能大于n_features")
    rng = np.random.default_rng(_random_state(random_state))
    return np.vstack([np.sort(rng.choice(features, size=size, replace=False)) for _ in range(learners)])


def _label_vector(y: np.ndarray) -> np.ndarray:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空一维数组")
    if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
        raise ValueError("y不能包含非有限值")
    if len(np.unique(y)) < 2:
        raise ValueError("标签扰动至少需要两个类别")
    return y


def flipped_label_copies(y: np.ndarray, n_learners: int, *,
                         flip_fraction: float = 0.1,
                         random_state: int = 0) -> np.ndarray:
    """输出表示扰动：每份标签精确翻转round(n*fraction)个位置到其他类别。"""
    labels = _label_vector(y); learners = _positive_integer(n_learners, "n_learners")
    if (isinstance(flip_fraction, (bool, np.bool_)) or not isinstance(flip_fraction, Real)
            or not np.isfinite(flip_fraction) or not 0 <= flip_fraction <= 1):
        raise ValueError("flip_fraction必须位于[0,1]")
    rng = np.random.default_rng(_random_state(random_state)); classes = np.unique(labels)
    copies = np.tile(labels, (learners, 1)); n_flip = int(round(len(labels) * float(flip_fraction)))
    for learner in range(learners):
        indices = rng.choice(len(labels), size=n_flip, replace=False)
        for index in indices:
            alternatives = classes[classes != labels[index]]
            copies[learner, index] = rng.choice(alternatives)
    return copies


def random_parameter_seeds(n_learners: int, *, random_state: int = 0) -> np.ndarray:
    """算法参数扰动入口：为每个基学习器生成独立初始化种子。"""
    learners = _positive_integer(n_learners, "n_learners")
    rng = np.random.default_rng(_random_state(random_state))
    return rng.integers(0, np.iinfo(np.uint32).max, size=learners, dtype=np.uint32)


def diversity_perturbation_plan(n_samples: int, n_features: int, y: np.ndarray, *,
                                n_learners: int, subspace_size: int,
                                flip_fraction: float = 0.1,
                                random_state: int = 0) -> dict[str, np.ndarray]:
    """用一个主种子派生四类可复现扰动，便于审计每个学习器得到的差异。"""
    if len(_label_vector(y)) != n_samples:
        raise ValueError("y长度必须等于n_samples")
    root_rng = np.random.default_rng(_random_state(random_state))
    seeds = root_rng.integers(0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32)
    return {
        "sample_indices": bootstrap_index_matrix(n_samples, n_learners, random_state=int(seeds[0])),
        "feature_subspaces": random_feature_subspaces(n_features, n_learners, subspace_size,
                                                        random_state=int(seeds[1])),
        "perturbed_labels": flipped_label_copies(y, n_learners, flip_fraction=flip_fraction,
                                                   random_state=int(seeds[2])),
        "parameter_seeds": random_parameter_seeds(n_learners, random_state=int(seeds[3])),
    }
