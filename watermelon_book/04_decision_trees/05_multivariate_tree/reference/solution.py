"""参考实现：确定性坐标搜索的多变量线性划分树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]


def _validate_xy(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y必须是一维且样本数与X一致")
    if not np.all(np.isfinite(X)) or not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("X和y必须包含有限数值")


def _validate_weights(weights: np.ndarray, n_features: int) -> None:
    if not isinstance(weights, np.ndarray) or weights.shape != (n_features,):
        raise ValueError("weights必须具有形状(n_features,)")
    if not np.all(np.isfinite(weights)) or np.linalg.norm(weights) == 0:
        raise ValueError("weights必须是非零有限向量")


def _normalized(weights: np.ndarray) -> np.ndarray:
    result = weights.astype(float, copy=True) / np.linalg.norm(weights)
    first = np.flatnonzero(np.abs(result) > 1e-12)[0]
    if result[first] < 0:
        result *= -1
    return result


def _gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def projection_values(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限数值")
    _validate_weights(weights, X.shape[1])
    return X @ weights


def best_projection_threshold(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    min_samples_leaf: int = 1,
) -> tuple[float, float]:
    _validate_xy(X, y)
    _validate_weights(weights, X.shape[1])
    if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf必须是正整数")
    projections = projection_values(X, _normalized(weights))
    values = np.unique(projections)
    if values.size < 2:
        raise ValueError("当前方向无法产生两个投影取值")
    thresholds = (values[:-1] + values[1:]) / 2.0
    best_threshold = None
    best_gini = np.inf
    for threshold in thresholds:
        left = projections <= threshold
        if left.sum() < min_samples_leaf or (~left).sum() < min_samples_leaf:
            continue
        score = left.mean() * _gini(y[left]) + (~left).mean() * _gini(y[~left])
        if score < best_gini - 1e-12:
            best_threshold = float(threshold)
            best_gini = float(score)
    if best_threshold is None:
        raise ValueError("没有满足min_samples_leaf的投影阈值")
    return best_threshold, best_gini


def coordinate_search_split(
    X: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    *,
    min_samples_leaf: int = 1,
    initial_step: float = 1.0,
    max_rounds: int = 8,
) -> tuple[np.ndarray, float, float]:
    _validate_xy(X, y)
    _validate_weights(initial_weights, X.shape[1])
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("initial_step必须是正有限数值")
    if not isinstance(max_rounds, (int, np.integer)) or max_rounds < 0:
        raise ValueError("max_rounds必须是非负整数")
    weights = _normalized(initial_weights)
    threshold, score = best_projection_threshold(
        X, y, weights, min_samples_leaf=min_samples_leaf
    )
    step = float(initial_step)
    for _ in range(max_rounds):
        improved = False
        for feature_index in range(X.shape[1]):
            for direction in (1.0, -1.0):
                candidate = weights.copy()
                candidate[feature_index] += direction * step
                if np.linalg.norm(candidate) <= 1e-12:
                    continue
                candidate = _normalized(candidate)
                candidate_threshold, candidate_score = best_projection_threshold(
                    X, y, candidate, min_samples_leaf=min_samples_leaf
                )
                if candidate_score < score - 1e-12:
                    weights, threshold, score = candidate, candidate_threshold, candidate_score
                    improved = True
        if not improved:
            step *= 0.5
    return weights, threshold, score


def _initial_directions(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    directions = [np.eye(X.shape[1])[index] for index in range(X.shape[1])]
    classes = np.unique(y)
    means = {label: X[y == label].mean(axis=0) for label in classes}
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1 :]:
            difference = means[right] - means[left]
            if np.linalg.norm(difference) > 1e-12:
                directions.append(_normalized(difference))
    return directions


def find_linear_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    min_samples_leaf: int = 1,
    max_search_rounds: int = 8,
) -> tuple[np.ndarray, float, float]:
    _validate_xy(X, y)
    best = None
    for direction in _initial_directions(X, y):
        try:
            candidate = coordinate_search_split(
                X,
                y,
                direction,
                min_samples_leaf=min_samples_leaf,
                max_rounds=max_search_rounds,
            )
        except ValueError:
            continue
        if best is None or candidate[2] < best[2] - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("无法找到有效线性划分")
    return best


def best_axis_parallel_split(
    X: np.ndarray, y: np.ndarray, *, min_samples_leaf: int = 1
) -> tuple[np.ndarray, float, float]:
    _validate_xy(X, y)
    best = None
    for index in range(X.shape[1]):
        direction = np.eye(X.shape[1])[index]
        try:
            threshold, score = best_projection_threshold(
                X, y, direction, min_samples_leaf=min_samples_leaf
            )
        except ValueError:
            continue
        candidate = (direction, threshold, score)
        if best is None or score < best[2] - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("无法找到有效轴平行划分")
    return best


def _probabilities(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.array([np.mean(y == label) for label in classes])


def _leaf(y: np.ndarray, classes: np.ndarray) -> Tree:
    probabilities = _probabilities(y, classes)
    return {
        "is_leaf": True,
        "prediction": classes[np.argmax(probabilities)].item(),
        "class_probabilities": probabilities,
        "weights": None,
        "threshold": None,
        "left": None,
        "right": None,
    }


def _build(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_search_rounds: int,
) -> Tree:
    if np.unique(y).size == 1 or depth >= max_depth or X.shape[0] < 2 * min_samples_leaf:
        return _leaf(y, classes)
    try:
        weights, threshold, score = find_linear_split(
            X,
            y,
            min_samples_leaf=min_samples_leaf,
            max_search_rounds=max_search_rounds,
        )
    except ValueError:
        return _leaf(y, classes)
    left = projection_values(X, weights) <= threshold
    probabilities = _probabilities(y, classes)
    return {
        "is_leaf": False,
        "prediction": classes[np.argmax(probabilities)].item(),
        "class_probabilities": probabilities,
        "weights": weights,
        "threshold": threshold,
        "weighted_gini": score,
        "left": _build(
            X[left], y[left], classes, depth + 1, max_depth, min_samples_leaf, max_search_rounds
        ),
        "right": _build(
            X[~left], y[~left], classes, depth + 1, max_depth, min_samples_leaf, max_search_rounds
        ),
    }


def fit_multivariate_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    max_search_rounds: int = 8,
) -> Tree:
    _validate_xy(X, y)
    if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
        raise ValueError("max_depth必须是非负整数")
    if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf必须是正整数")
    if not isinstance(max_search_rounds, (int, np.integer)) or max_search_rounds < 0:
        raise ValueError("max_search_rounds必须是非负整数")
    return _build(
        X, y, np.unique(y), 0, int(max_depth), int(min_samples_leaf), int(max_search_rounds)
    )


def predict_multivariate_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    if not isinstance(tree, dict) or "is_leaf" not in tree:
        raise ValueError("tree不是有效树节点")
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限数值")
    results = []
    for row in X:
        node = tree
        while not node["is_leaf"]:
            if row.size != node["weights"].size:
                raise ValueError("预测特征数与树不一致")
            node = node["left"] if row @ node["weights"] <= node["threshold"] else node["right"]
        results.append(node["prediction"])
    return np.asarray(results)


def count_tree_nodes(tree: Tree) -> int:
    if tree["is_leaf"]:
        return 1
    return 1 + count_tree_nodes(tree["left"]) + count_tree_nodes(tree["right"])
