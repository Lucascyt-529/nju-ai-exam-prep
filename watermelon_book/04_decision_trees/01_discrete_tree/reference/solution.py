"""参考实现：离散特征决策树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]
VALID_CRITERIA = {"information_gain", "gain_ratio", "gini"}


def _validate_labels(y: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空一维数组")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须包含有限数值标签")


def _validate_feature_and_labels(feature: np.ndarray, y: np.ndarray) -> None:
    _validate_labels(y)
    if not isinstance(feature, np.ndarray) or feature.ndim != 1 or feature.shape[0] != y.size:
        raise ValueError("feature必须是一维且样本数与y一致")
    if not np.issubdtype(feature.dtype, np.number) or not np.all(np.isfinite(feature)):
        raise ValueError("feature必须包含有限离散数值")


def _validate_training_data(X: np.ndarray, y: np.ndarray) -> None:
    _validate_labels(y)
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] != y.size
        or X.shape[1] == 0
    ):
        raise ValueError("X必须是与y样本数一致的非空二维特征数组")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限离散数值")


def entropy(y: np.ndarray) -> float:
    _validate_labels(y)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def gini(y: np.ndarray) -> float:
    _validate_labels(y)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def majority_label(y: np.ndarray):
    _validate_labels(y)
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)].item()


def _weighted_child_impurity(
    feature: np.ndarray, y: np.ndarray, impurity_function
) -> float:
    total = y.size
    result = 0.0
    for value in np.unique(feature):
        selected = feature == value
        result += selected.sum() / total * impurity_function(y[selected])
    return float(result)


def information_gain(feature: np.ndarray, y: np.ndarray) -> float:
    _validate_feature_and_labels(feature, y)
    return entropy(y) - _weighted_child_impurity(feature, y, entropy)


def gain_ratio(feature: np.ndarray, y: np.ndarray) -> float:
    _validate_feature_and_labels(feature, y)
    intrinsic_value = entropy(feature)
    if intrinsic_value == 0:
        return 0.0
    return information_gain(feature, y) / intrinsic_value


def gini_index(feature: np.ndarray, y: np.ndarray) -> float:
    _validate_feature_and_labels(feature, y)
    return _weighted_child_impurity(feature, y, gini)


def choose_best_feature(
    X: np.ndarray,
    y: np.ndarray,
    available_features: np.ndarray,
    *,
    criterion: str = "information_gain",
) -> tuple[int, float]:
    _validate_training_data(X, y)
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion必须属于{sorted(VALID_CRITERIA)}")
    if (
        not isinstance(available_features, np.ndarray)
        or available_features.ndim != 1
        or available_features.size == 0
        or not np.issubdtype(available_features.dtype, np.integer)
        or np.any((available_features < 0) | (available_features >= X.shape[1]))
        or np.unique(available_features).size != available_features.size
    ):
        raise ValueError("available_features必须是不重复的有效特征下标")

    candidates = np.sort(available_features)
    candidates = np.array(
        [index for index in candidates if np.unique(X[:, index]).size > 1], dtype=int
    )
    if candidates.size == 0:
        raise ValueError("当前子集中没有可产生分支的特征")

    if criterion == "information_gain":
        scores = np.array([information_gain(X[:, index], y) for index in candidates])
        best_position = int(np.argmax(scores))
    elif criterion == "gini":
        scores = np.array([gini_index(X[:, index], y) for index in candidates])
        best_position = int(np.argmin(scores))
    else:
        gains = np.array([information_gain(X[:, index], y) for index in candidates])
        ratios = np.array([gain_ratio(X[:, index], y) for index in candidates])
        eligible = gains >= gains.mean() - 1e-12
        eligible_positions = np.flatnonzero(eligible)
        best_position = int(eligible_positions[np.argmax(ratios[eligible])])
        scores = ratios
    return int(candidates[best_position]), float(scores[best_position])


def _leaf(prediction) -> Tree:
    return {
        "is_leaf": True,
        "prediction": prediction,
        "feature_index": None,
        "children": {},
    }


def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    available_features: np.ndarray,
    feature_domains: list[np.ndarray],
    criterion: str,
) -> Tree:
    prediction = majority_label(y)
    if np.unique(y).size == 1 or available_features.size == 0:
        return _leaf(prediction)

    variable_features = np.array(
        [index for index in available_features if np.unique(X[:, index]).size > 1],
        dtype=int,
    )
    if variable_features.size == 0:
        return _leaf(prediction)

    feature_index, score = choose_best_feature(
        X, y, variable_features, criterion=criterion
    )
    remaining = available_features[available_features != feature_index]
    node: Tree = {
        "is_leaf": False,
        "prediction": prediction,
        "feature_index": feature_index,
        "criterion_score": score,
        "children": {},
    }
    for value in feature_domains[feature_index]:
        selected = X[:, feature_index] == value
        key = value.item()
        if not np.any(selected):
            node["children"][key] = _leaf(prediction)
        else:
            node["children"][key] = _build_tree(
                X[selected],
                y[selected],
                remaining,
                feature_domains,
                criterion,
            )
    return node


def fit_discrete_tree(
    X: np.ndarray, y: np.ndarray, *, criterion: str = "information_gain"
) -> Tree:
    _validate_training_data(X, y)
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion必须属于{sorted(VALID_CRITERIA)}")
    feature_domains = [np.unique(X[:, index]) for index in range(X.shape[1])]
    available_features = np.arange(X.shape[1], dtype=int)
    return _build_tree(X, y, available_features, feature_domains, criterion)


def _validate_prediction_data(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限离散数值")


def _predict_one(tree: Tree, row: np.ndarray):
    node = tree
    while not node["is_leaf"]:
        feature_index = node["feature_index"]
        if feature_index >= row.size:
            raise ValueError("预测X的特征数少于树所需特征数")
        value = row[feature_index].item()
        child = node["children"].get(value)
        if child is None:
            return node["prediction"]
        node = child
    return node["prediction"]


def predict_discrete_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    _validate_prediction_data(X)
    if not isinstance(tree, dict) or "is_leaf" not in tree or "prediction" not in tree:
        raise ValueError("tree不是有效的参考实现树节点")
    return np.asarray([_predict_one(tree, row) for row in X])
