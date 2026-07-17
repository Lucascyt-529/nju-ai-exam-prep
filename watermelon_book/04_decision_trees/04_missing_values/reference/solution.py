"""参考实现：带缺失值和样本权重的离散决策树。"""

from typing import Any

import numpy as np


Model = dict[str, Any]
Node = dict[str, Any]
VALID_FEATURE_TYPES = {"discrete", "continuous"}


def _validate_y_and_weights(y: np.ndarray, sample_weights: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空一维数组")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须包含有限数值标签")
    if (
        not isinstance(sample_weights, np.ndarray)
        or sample_weights.shape != y.shape
        or not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights < 0)
        or sample_weights.sum() <= 0
    ):
        raise ValueError("sample_weights必须是同形状非负有限向量且总权重大于0")


def _validate_feature(feature: np.ndarray, size: int) -> None:
    if not isinstance(feature, np.ndarray) or feature.ndim != 1 or feature.size != size:
        raise ValueError("feature必须是一维且样本数匹配")
    if not np.issubdtype(feature.dtype, np.number) or np.any(np.isinf(feature)):
        raise ValueError("feature只能包含有限数值或np.nan")


def weighted_entropy(y: np.ndarray, sample_weights: np.ndarray) -> float:
    _validate_y_and_weights(y, sample_weights)
    total = sample_weights.sum()
    probabilities = np.array(
        [sample_weights[y == label].sum() / total for label in np.unique(y)]
    )
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))


def branch_weight_plan(
    feature: np.ndarray, sample_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy_y = np.zeros(sample_weights.shape, dtype=int)
    _validate_y_and_weights(dummy_y, sample_weights)
    _validate_feature(feature, sample_weights.size)
    known = ~np.isnan(feature) & (sample_weights > 0)
    missing = np.isnan(feature)
    values = np.unique(feature[known])
    if values.size < 2:
        raise ValueError("已知属性值必须至少包含两个分支")
    known_masses = np.array(
        [sample_weights[known & (feature == value)].sum() for value in values]
    )
    if known_masses.sum() <= 0 or np.any(known_masses <= 0):
        raise ValueError("每个已知分支必须具有正权重")
    proportions = known_masses / known_masses.sum()
    plan = np.zeros((feature.size, values.size), dtype=float)
    for column, value in enumerate(values):
        plan[known & (feature == value), column] = sample_weights[
            known & (feature == value)
        ]
        plan[missing, column] = sample_weights[missing] * proportions[column]
    return values, proportions, plan


def missing_information_gain(
    feature: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> float:
    weights = np.ones(y.shape, dtype=float) if sample_weights is None else sample_weights
    _validate_y_and_weights(y, weights)
    _validate_feature(feature, y.size)
    known = ~np.isnan(feature)
    known_weights = weights[known]
    if known_weights.sum() <= 0 or np.unique(feature[known & (weights > 0)]).size < 2:
        raise ValueError("非缺失正权重样本必须至少包含两个属性值")
    known_ratio = known_weights.sum() / weights.sum()
    parent_entropy = weighted_entropy(y[known], known_weights)
    conditional = 0.0
    for value in np.unique(feature[known & (weights > 0)]):
        selected = known & (feature == value)
        branch_weight = weights[selected].sum()
        conditional += branch_weight / known_weights.sum() * weighted_entropy(
            y[selected], weights[selected]
        )
    return float(known_ratio * (parent_entropy - conditional))


def continuous_missing_information_gain(
    feature: np.ndarray,
    y: np.ndarray,
    threshold: float,
    sample_weights: np.ndarray | None = None,
) -> float:
    """只在非缺失样本上计算连续二分增益，再乘有效权重比例。"""
    weights = np.ones(y.shape, dtype=float) if sample_weights is None else sample_weights
    _validate_y_and_weights(y, weights); _validate_feature(feature, y.size)
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold必须是有限标量")
    known = ~np.isnan(feature) & (weights > 0)
    left = known & (feature <= float(threshold)); right = known & ~left
    if weights[left].sum() <= 0 or weights[right].sum() <= 0:
        raise ValueError("threshold必须把非缺失正权重样本分成两个分支")
    known_weight = weights[known].sum(); known_ratio = known_weight / weights.sum()
    conditional = (
        weights[left].sum() / known_weight * weighted_entropy(y[left], weights[left])
        + weights[right].sum() / known_weight * weighted_entropy(y[right], weights[right])
    )
    return float(known_ratio * (weighted_entropy(y[known], weights[known]) - conditional))


def best_continuous_missing_split(
    feature: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    weights = np.ones(y.shape, dtype=float) if sample_weights is None else sample_weights
    _validate_y_and_weights(y, weights); _validate_feature(feature, y.size)
    known_values = np.unique(feature[~np.isnan(feature) & (weights > 0)])
    if known_values.size < 2:
        raise ValueError("非缺失正权重样本必须至少包含两个连续取值")
    thresholds = (known_values[:-1] + known_values[1:]) / 2.0
    gains = np.array([
        continuous_missing_information_gain(feature, y, threshold, weights)
        for threshold in thresholds
    ])
    position = int(np.argmax(gains))
    return float(thresholds[position]), float(gains[position])


def continuous_branch_weight_plan(
    feature: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回左右分支比例和逐样本权重计划，列0/1分别为left/right。"""
    dummy_y = np.zeros(sample_weights.shape, dtype=int)
    _validate_y_and_weights(dummy_y, sample_weights); _validate_feature(feature, sample_weights.size)
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold必须是有限标量")
    known = ~np.isnan(feature) & (sample_weights > 0); missing = np.isnan(feature)
    left = known & (feature <= float(threshold)); right = known & ~left
    masses = np.array([sample_weights[left].sum(), sample_weights[right].sum()])
    if np.any(masses <= 0):
        raise ValueError("threshold必须产生两个正权重已知分支")
    proportions = masses / masses.sum(); plan = np.zeros((feature.size, 2), dtype=float)
    plan[left, 0] = sample_weights[left]; plan[right, 1] = sample_weights[right]
    plan[missing] = sample_weights[missing, None] * proportions[None, :]
    return proportions, plan


def _class_probabilities(
    y: np.ndarray, weights: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    total = weights.sum()
    return np.array([weights[y == label].sum() / total for label in classes])


def _leaf(y: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> Node:
    return {
        "is_leaf": True,
        "prediction": classes[np.argmax(_class_probabilities(y, weights, classes))].item(),
        "class_probabilities": _class_probabilities(y, weights, classes),
        "feature_index": None,
        "children": {},
    }


def _build(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    available_features: tuple[int, ...],
    classes: np.ndarray,
    feature_types: tuple[str, ...],
) -> Node:
    positive = weights > 0
    if np.unique(y[positive]).size == 1 or not available_features:
        return _leaf(y, weights, classes)
    candidates = []
    for index in available_features:
        known_positive = ~np.isnan(X[:, index]) & positive
        if np.unique(X[known_positive, index]).size < 2:
            continue
        if feature_types[index] == "discrete":
            candidates.append((missing_information_gain(X[:, index], y, weights), index, None))
        else:
            threshold, gain = best_continuous_missing_split(X[:, index], y, weights)
            candidates.append((gain, index, threshold))
    if not candidates:
        return _leaf(y, weights, classes)
    gain, feature_index, threshold = max(
        candidates,
        key=lambda item: (item[0], -item[1], -(item[2] if item[2] is not None else 0.0)),
    )
    split_type = feature_types[feature_index]
    if split_type == "discrete":
        values, proportions, plan = branch_weight_plan(X[:, feature_index], weights)
        branch_keys: tuple[Any, ...] = tuple(value.item() for value in values)
    else:
        proportions, plan = continuous_branch_weight_plan(
            X[:, feature_index], weights, threshold
        )
        values = None; branch_keys = ("left", "right")
    node: Node = {
        "is_leaf": False,
        "prediction": classes[np.argmax(_class_probabilities(y, weights, classes))].item(),
        "class_probabilities": _class_probabilities(y, weights, classes),
        "feature_index": feature_index,
        "split_type": split_type,
        "threshold": threshold,
        "information_gain": gain,
        "branch_values": values,
        "branch_keys": branch_keys,
        "branch_probabilities": proportions,
        "children": {},
    }
    remaining = (
        tuple(value for value in available_features if value != feature_index)
        if split_type == "discrete" else available_features
    )
    for column, branch_key in enumerate(branch_keys):
        branch_weights = plan[:, column]
        selected = branch_weights > 0
        node["children"][branch_key] = _build(
            X[selected],
            y[selected],
            branch_weights[selected],
            remaining,
            classes,
            feature_types,
        )
    return node


def fit_missing_value_tree(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
    *,
    feature_types: list[str] | None = None,
) -> Model:
    weights = np.ones(y.shape, dtype=float) if sample_weights is None else sample_weights.copy()
    _validate_y_and_weights(y, weights)
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] != y.size or X.shape[1] == 0:
        raise ValueError("X必须是与y样本数一致的非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or np.any(np.isinf(X)):
        raise ValueError("X只能包含有限数值或np.nan")
    types = ["discrete"] * X.shape[1] if feature_types is None else feature_types
    if (not isinstance(types, list) or len(types) != X.shape[1]
            or any(value not in VALID_FEATURE_TYPES for value in types)):
        raise ValueError(f"feature_types必须逐列属于{sorted(VALID_FEATURE_TYPES)}")
    classes = np.unique(y)
    type_tuple = tuple(types)
    root = _build(X, y, weights, tuple(range(X.shape[1])), classes, type_tuple)
    return {"classes": classes, "n_features": X.shape[1], "feature_types": type_tuple, "root": root}


def _predict_distribution(node: Node, row: np.ndarray) -> np.ndarray:
    if node["is_leaf"]:
        return node["class_probabilities"]
    value = row[node["feature_index"]]
    if np.isnan(value):
        result = np.zeros_like(node["class_probabilities"])
        for branch_key, proportion in zip(
            node["branch_keys"], node["branch_probabilities"], strict=True
        ):
            result += proportion * _predict_distribution(
                node["children"][branch_key], row
            )
        return result
    if node["split_type"] == "continuous":
        branch_key = "left" if value <= node["threshold"] else "right"
        return _predict_distribution(node["children"][branch_key], row)
    child = node["children"].get(value.item())
    return node["class_probabilities"] if child is None else _predict_distribution(child, row)


def predict_proba_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    if not isinstance(model, dict) or not {"classes", "n_features", "root"}.issubset(model):
        raise ValueError("model不是有效的缺失值树模型")
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] != model["n_features"]
    ):
        raise ValueError("X必须是特征数匹配的非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or np.any(np.isinf(X)):
        raise ValueError("X只能包含有限数值或np.nan")
    probabilities = np.vstack([_predict_distribution(model["root"], row) for row in X])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def predict_missing_tree(model: Model, X: np.ndarray) -> np.ndarray:
    probabilities = predict_proba_missing_tree(model, X)
    return model["classes"][np.argmax(probabilities, axis=1)]
