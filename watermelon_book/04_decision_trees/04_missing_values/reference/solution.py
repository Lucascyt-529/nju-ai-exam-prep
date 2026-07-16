"""参考实现：带缺失值和样本权重的离散决策树。"""

from typing import Any

import numpy as np


Model = dict[str, Any]
Node = dict[str, Any]


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
) -> Node:
    positive = weights > 0
    if np.unique(y[positive]).size == 1 or not available_features:
        return _leaf(y, weights, classes)
    candidates = []
    for index in available_features:
        known_positive = ~np.isnan(X[:, index]) & positive
        if np.unique(X[known_positive, index]).size >= 2:
            candidates.append((missing_information_gain(X[:, index], y, weights), index))
    if not candidates:
        return _leaf(y, weights, classes)
    gain, feature_index = max(candidates, key=lambda item: (item[0], -item[1]))
    values, proportions, plan = branch_weight_plan(X[:, feature_index], weights)
    node: Node = {
        "is_leaf": False,
        "prediction": classes[np.argmax(_class_probabilities(y, weights, classes))].item(),
        "class_probabilities": _class_probabilities(y, weights, classes),
        "feature_index": feature_index,
        "information_gain": gain,
        "branch_values": values,
        "branch_probabilities": proportions,
        "children": {},
    }
    remaining = tuple(value for value in available_features if value != feature_index)
    for column, value in enumerate(values):
        branch_weights = plan[:, column]
        selected = branch_weights > 0
        node["children"][value.item()] = _build(
            X[selected],
            y[selected],
            branch_weights[selected],
            remaining,
            classes,
        )
    return node


def fit_missing_value_tree(
    X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None
) -> Model:
    weights = np.ones(y.shape, dtype=float) if sample_weights is None else sample_weights.copy()
    _validate_y_and_weights(y, weights)
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] != y.size or X.shape[1] == 0:
        raise ValueError("X必须是与y样本数一致的非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or np.any(np.isinf(X)):
        raise ValueError("X只能包含有限数值或np.nan")
    classes = np.unique(y)
    root = _build(X, y, weights, tuple(range(X.shape[1])), classes)
    return {"classes": classes, "n_features": X.shape[1], "root": root}


def _predict_distribution(node: Node, row: np.ndarray) -> np.ndarray:
    if node["is_leaf"]:
        return node["class_probabilities"]
    value = row[node["feature_index"]]
    if np.isnan(value):
        result = np.zeros_like(node["class_probabilities"])
        for branch_value, proportion in zip(
            node["branch_values"], node["branch_probabilities"]
        ):
            result += proportion * _predict_distribution(
                node["children"][branch_value.item()], row
            )
        return result
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
