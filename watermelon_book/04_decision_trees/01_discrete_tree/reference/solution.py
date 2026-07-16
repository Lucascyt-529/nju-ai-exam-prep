"""参考实现：离散特征决策树与验证集剪枝。"""

import copy
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


def count_tree_nodes(tree: Tree) -> int:
    if not isinstance(tree, dict) or "is_leaf" not in tree or "children" not in tree:
        raise ValueError("tree不是有效的参考实现树节点")
    if tree["is_leaf"]:
        return 1
    return 1 + sum(count_tree_nodes(child) for child in tree["children"].values())


def _validate_validation_data(
    X_validation: np.ndarray, y_validation: np.ndarray, n_features: int
) -> None:
    _validate_training_data(X_validation, y_validation)
    if X_validation.shape[1] != n_features:
        raise ValueError("训练集与验证集特征数必须一致")


def _accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predictions == y))


def _build_prepruned_tree(
    X: np.ndarray,
    y: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    available_features: np.ndarray,
    feature_domains: list[np.ndarray],
    criterion: str,
) -> Tree:
    prediction = majority_label(y)
    if np.unique(y).size == 1 or available_features.size == 0:
        return _leaf(prediction)
    if y_validation.size == 0:
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
    leaf_predictions = np.full(y_validation.shape, prediction)
    split_predictions = np.full(y_validation.shape, prediction)
    for value in feature_domains[feature_index]:
        train_selected = X[:, feature_index] == value
        validation_selected = X_validation[:, feature_index] == value
        if np.any(train_selected):
            split_predictions[validation_selected] = majority_label(y[train_selected])

    if _accuracy(split_predictions, y_validation) <= _accuracy(
        leaf_predictions, y_validation
    ):
        return _leaf(prediction)

    remaining = available_features[available_features != feature_index]
    node: Tree = {
        "is_leaf": False,
        "prediction": prediction,
        "feature_index": feature_index,
        "criterion_score": score,
        "children": {},
    }
    for value in feature_domains[feature_index]:
        train_selected = X[:, feature_index] == value
        validation_selected = X_validation[:, feature_index] == value
        key = value.item()
        if not np.any(train_selected):
            node["children"][key] = _leaf(prediction)
        else:
            node["children"][key] = _build_prepruned_tree(
                X[train_selected],
                y[train_selected],
                X_validation[validation_selected],
                y_validation[validation_selected],
                remaining,
                feature_domains,
                criterion,
            )
    return node


def fit_prepruned_tree(
    X: np.ndarray,
    y: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    criterion: str = "information_gain",
) -> Tree:
    """仅用训练集选择划分，用验证集决定是否保留划分。"""
    _validate_training_data(X, y)
    _validate_validation_data(X_validation, y_validation, X.shape[1])
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion必须属于{sorted(VALID_CRITERIA)}")
    feature_domains = [np.unique(X[:, index]) for index in range(X.shape[1])]
    available_features = np.arange(X.shape[1], dtype=int)
    return _build_prepruned_tree(
        X,
        y,
        X_validation,
        y_validation,
        available_features,
        feature_domains,
        criterion,
    )


def _post_prune_node(
    node: Tree, X_validation: np.ndarray, y_validation: np.ndarray
) -> Tree:
    if node["is_leaf"]:
        return node
    if y_validation.size == 0:
        return _leaf(node["prediction"])

    feature_index = node["feature_index"]
    for value, child in list(node["children"].items()):
        selected = X_validation[:, feature_index] == value
        node["children"][value] = _post_prune_node(
            child, X_validation[selected], y_validation[selected]
        )

    subtree_predictions = np.asarray(
        [_predict_one(node, row) for row in X_validation]
    )
    leaf_predictions = np.full(y_validation.shape, node["prediction"])
    if _accuracy(leaf_predictions, y_validation) >= _accuracy(
        subtree_predictions, y_validation
    ):
        return _leaf(node["prediction"])
    return node


def post_prune_tree(
    tree: Tree, X_validation: np.ndarray, y_validation: np.ndarray
) -> Tree:
    """自底向上后剪枝，准确率相同时选择更小的叶节点。"""
    _validate_prediction_data(X_validation)
    _validate_labels(y_validation)
    if X_validation.shape[0] != y_validation.size:
        raise ValueError("验证集X与y样本数必须一致")
    if not isinstance(tree, dict) or "is_leaf" not in tree or "prediction" not in tree:
        raise ValueError("tree不是有效的参考实现树节点")
    copied_tree = copy.deepcopy(tree)
    return _post_prune_node(copied_tree, X_validation, y_validation)
