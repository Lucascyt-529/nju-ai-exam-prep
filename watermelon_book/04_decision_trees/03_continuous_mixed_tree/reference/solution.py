"""参考实现：连续值与离散/连续混合决策树。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]
VALID_CRITERIA = {"information_gain", "gini"}
VALID_FEATURE_TYPES = {"continuous", "discrete"}


def _validate_y(y: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
        raise ValueError("y必须是非空一维数组")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须包含有限数值标签")


def _validate_feature(feature: np.ndarray, y: np.ndarray | None = None) -> None:
    if not isinstance(feature, np.ndarray) or feature.ndim != 1 or feature.size == 0:
        raise ValueError("feature必须是非空一维数组")
    if not np.issubdtype(feature.dtype, np.number) or not np.all(np.isfinite(feature)):
        raise ValueError("feature必须包含有限数值")
    if y is not None:
        _validate_y(y)
        if feature.size != y.size:
            raise ValueError("feature与y样本数必须一致")


def _entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def _majority(y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)].item()


def candidate_thresholds(feature: np.ndarray) -> np.ndarray:
    _validate_feature(feature)
    values = np.unique(feature.astype(float))
    if values.size < 2:
        return np.empty(0, dtype=float)
    return (values[:-1] + values[1:]) / 2.0


def continuous_split_score(
    feature: np.ndarray,
    y: np.ndarray,
    threshold: float,
    *,
    criterion: str = "information_gain",
) -> float:
    _validate_feature(feature, y)
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion必须属于{sorted(VALID_CRITERIA)}")
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("threshold必须是有限标量")
    left = feature <= float(threshold)
    if not np.any(left) or np.all(left):
        raise ValueError("threshold必须产生两个非空子集")
    impurity = _entropy if criterion == "information_gain" else _gini
    weighted = left.mean() * impurity(y[left]) + (~left).mean() * impurity(y[~left])
    return _entropy(y) - weighted if criterion == "information_gain" else weighted


def best_continuous_split(
    feature: np.ndarray, y: np.ndarray, *, criterion: str = "information_gain"
) -> tuple[float, float]:
    _validate_feature(feature, y)
    thresholds = candidate_thresholds(feature)
    if thresholds.size == 0:
        raise ValueError("连续特征只有一个取值，无法二分")
    scores = np.array(
        [continuous_split_score(feature, y, value, criterion=criterion) for value in thresholds]
    )
    position = int(np.argmax(scores) if criterion == "information_gain" else np.argmin(scores))
    return float(thresholds[position]), float(scores[position])


def _discrete_score(feature: np.ndarray, y: np.ndarray, criterion: str) -> float:
    impurity = _entropy if criterion == "information_gain" else _gini
    weighted = 0.0
    for value in np.unique(feature):
        selected = feature == value
        weighted += selected.mean() * impurity(y[selected])
    return _entropy(y) - weighted if criterion == "information_gain" else weighted


def _leaf(prediction) -> Tree:
    return {"is_leaf": True, "prediction": prediction, "feature_index": None, "children": {}}


def _build(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: tuple[str, ...],
    available_discrete: set[int],
    discrete_domains: dict[int, np.ndarray],
    criterion: str,
) -> Tree:
    prediction = _majority(y)
    if np.unique(y).size == 1:
        return _leaf(prediction)

    candidates = []
    for index, feature_type in enumerate(feature_types):
        if feature_type == "discrete":
            if index not in available_discrete or np.unique(X[:, index]).size < 2:
                continue
            candidates.append((index, None, _discrete_score(X[:, index], y, criterion)))
        elif np.unique(X[:, index]).size >= 2:
            threshold, score = best_continuous_split(X[:, index], y, criterion=criterion)
            candidates.append((index, threshold, score))
    if not candidates:
        return _leaf(prediction)

    if criterion == "information_gain":
        best = max(candidates, key=lambda item: (item[2], -item[0], -(item[1] or 0.0)))
    else:
        best = min(candidates, key=lambda item: (item[2], item[0], item[1] or 0.0))
    feature_index, threshold, score = best
    feature_type = feature_types[feature_index]
    node: Tree = {
        "is_leaf": False,
        "prediction": prediction,
        "feature_index": feature_index,
        "split_type": feature_type,
        "threshold": threshold,
        "criterion_score": score,
        "children": {},
    }

    if feature_type == "continuous":
        left = X[:, feature_index] <= threshold
        node["children"]["left"] = _build(
            X[left], y[left], feature_types, available_discrete.copy(), discrete_domains, criterion
        )
        node["children"]["right"] = _build(
            X[~left], y[~left], feature_types, available_discrete.copy(), discrete_domains, criterion
        )
    else:
        remaining = available_discrete - {feature_index}
        for value in discrete_domains[feature_index]:
            selected = X[:, feature_index] == value
            key = value.item()
            node["children"][key] = (
                _build(X[selected], y[selected], feature_types, remaining.copy(), discrete_domains, criterion)
                if np.any(selected)
                else _leaf(prediction)
            )
    return node


def fit_mixed_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: list[str],
    *,
    criterion: str = "information_gain",
) -> Tree:
    _validate_y(y)
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] != y.size or X.shape[1] == 0:
        raise ValueError("X必须是与y样本数一致的非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限数值")
    if not isinstance(feature_types, list) or len(feature_types) != X.shape[1]:
        raise ValueError("feature_types必须为每列提供类型")
    if any(value not in VALID_FEATURE_TYPES for value in feature_types):
        raise ValueError(f"特征类型必须属于{sorted(VALID_FEATURE_TYPES)}")
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion必须属于{sorted(VALID_CRITERIA)}")
    types = tuple(feature_types)
    discrete = {index for index, value in enumerate(types) if value == "discrete"}
    domains = {index: np.unique(X[:, index]) for index in discrete}
    return _build(X, y, types, discrete, domains, criterion)


def predict_mixed_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须包含有限数值")
    if not isinstance(tree, dict) or "is_leaf" not in tree:
        raise ValueError("tree不是有效树节点")
    results = []
    for row in X:
        node = tree
        while not node["is_leaf"]:
            index = node["feature_index"]
            if index >= row.size:
                raise ValueError("预测X特征数不足")
            if node["split_type"] == "continuous":
                branch = "left" if row[index] <= node["threshold"] else "right"
                node = node["children"][branch]
            else:
                child = node["children"].get(row[index].item())
                if child is None:
                    results.append(node["prediction"])
                    break
                node = child
        else:
            results.append(node["prediction"])
    return np.asarray(results)
