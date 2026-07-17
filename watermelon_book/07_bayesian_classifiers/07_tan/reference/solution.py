"""参考实现：离散TAN结构学习、参数估计与分类。"""

from numbers import Real
import numpy as np


def _validate_training(X: np.ndarray, y: np.ndarray) -> None:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not np.issubdtype(X.dtype, np.integer)):
        raise ValueError("X必须是非空整数二维数组")
    if (not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != len(X)
            or not np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 2):
        raise ValueError("y必须是一维整数数组且至少含两个类别")


def _validate_feature_pair(n_features: int, feature_i: int, feature_j: int) -> None:
    for value in (feature_i, feature_j):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("特征下标必须是整数")
    if not (0 <= feature_i < n_features and 0 <= feature_j < n_features) or feature_i == feature_j:
        raise ValueError("特征下标越界或重复")


def conditional_mutual_information(X: np.ndarray, y: np.ndarray,
                                   feature_i: int, feature_j: int) -> float:
    _validate_training(X, y); _validate_feature_pair(X.shape[1], feature_i, feature_j)
    total = 0.0
    for label in np.unique(y):
        subset = X[y == label]
        class_probability = len(subset) / len(X)
        values_i, inverse_i = np.unique(subset[:, feature_i], return_inverse=True)
        values_j, inverse_j = np.unique(subset[:, feature_j], return_inverse=True)
        joint = np.zeros((len(values_i), len(values_j)), dtype=float)
        np.add.at(joint, (inverse_i, inverse_j), 1.0)
        joint /= len(subset)
        marginal_i = joint.sum(axis=1, keepdims=True)
        marginal_j = joint.sum(axis=0, keepdims=True)
        positive = joint > 0
        ratio = np.zeros_like(joint)
        ratio[positive] = joint[positive] / (marginal_i @ marginal_j)[positive]
        total += class_probability * float(np.sum(joint[positive] * np.log(ratio[positive])))
    return float(total)


def conditional_mi_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    _validate_training(X, y)
    matrix = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for feature_i in range(X.shape[1]):
        for feature_j in range(feature_i + 1, X.shape[1]):
            value = conditional_mutual_information(X, y, feature_i, feature_j)
            matrix[feature_i, feature_j] = matrix[feature_j, feature_i] = value
    return matrix


def maximum_spanning_tree(weights: np.ndarray, *, root: int = 0) -> np.ndarray:
    if (not isinstance(weights, np.ndarray) or weights.ndim != 2
            or weights.shape[0] != weights.shape[1] or weights.shape[0] == 0
            or not np.issubdtype(weights.dtype, np.number) or not np.all(np.isfinite(weights))
            or not np.allclose(weights, weights.T) or not np.allclose(np.diag(weights), 0.0)):
        raise ValueError("weights必须是对称、有限、零对角方阵")
    if (isinstance(root, (bool, np.bool_)) or not isinstance(root, (int, np.integer))
            or not 0 <= root < len(weights)):
        raise ValueError("root下标无效")
    parents = np.full(len(weights), -2, dtype=int); parents[root] = -1
    selected = {int(root)}
    while len(selected) < len(weights):
        best = None
        for parent in sorted(selected):
            for child in range(len(weights)):
                if child in selected:
                    continue
                key = (float(weights[parent, child]), -parent, -child)
                if best is None or key > best[0]:
                    best = (key, parent, child)
        _, parent, child = best
        parents[child] = parent; selected.add(child)
    return parents


def _index_maps(feature_values: tuple[np.ndarray, ...]) -> tuple[dict[int, int], ...]:
    return tuple({int(value): index for index, value in enumerate(values)} for values in feature_values)


def fit_tan(X: np.ndarray, y: np.ndarray, *, root: int = 0,
            alpha: float = 1.0) -> dict[str, object]:
    _validate_training(X, y)
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha必须是有限正数")
    weights = conditional_mi_matrix(X, y)
    parents = maximum_spanning_tree(weights, root=root)
    classes = np.unique(y)
    feature_values = tuple(np.unique(X[:, feature]) for feature in range(X.shape[1]))
    maps = _index_maps(feature_values)
    class_log_prior = np.empty(len(classes), dtype=float)
    log_probabilities: list[np.ndarray] = [np.empty(0)] * X.shape[1]
    for class_index, label in enumerate(classes):
        class_mask = y == label; class_count = int(np.sum(class_mask))
        class_log_prior[class_index] = np.log((class_count + alpha) / (len(y) + alpha * len(classes)))
        for feature in range(X.shape[1]):
            cardinality = len(feature_values[feature])
            if parents[feature] == -1:
                if class_index == 0:
                    log_probabilities[feature] = np.empty((len(classes), cardinality), dtype=float)
                counts = np.zeros(cardinality, dtype=float)
                for value in X[class_mask, feature]: counts[maps[feature][int(value)]] += 1
                log_probabilities[feature][class_index] = np.log((counts + alpha) / (class_count + alpha * cardinality))
            else:
                parent = int(parents[feature]); parent_cardinality = len(feature_values[parent])
                if class_index == 0:
                    log_probabilities[feature] = np.empty((len(classes), parent_cardinality, cardinality), dtype=float)
                counts = np.zeros((parent_cardinality, cardinality), dtype=float)
                for row in X[class_mask]:
                    counts[maps[parent][int(row[parent])], maps[feature][int(row[feature])]] += 1
                denominators = counts.sum(axis=1, keepdims=True) + alpha * cardinality
                log_probabilities[feature][class_index] = np.log((counts + alpha) / denominators)
    return {"classes": classes.copy(), "feature_values": tuple(values.copy() for values in feature_values),
            "parents": parents, "root": int(root), "weights": weights,
            "class_log_prior": class_log_prior, "log_probabilities": tuple(log_probabilities)}


def _validate_model(model: dict[str, object]) -> None:
    required = {"classes", "feature_values", "parents", "root", "weights", "class_log_prior", "log_probabilities"}
    if not isinstance(model, dict) or set(model) != required:
        raise ValueError("model键集合无效")


def tan_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model)
    n_features = len(model["feature_values"])
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != n_features
            or not np.issubdtype(X.dtype, np.integer)):
        raise ValueError("X必须是列数匹配的整数二维数组")
    maps = _index_maps(model["feature_values"])
    scores = np.tile(model["class_log_prior"], (len(X), 1))
    for row_index, row in enumerate(X):
        try:
            indices = [maps[feature][int(row[feature])] for feature in range(n_features)]
        except KeyError as error:
            raise ValueError("预测样本含训练时未见的属性值") from error
        for feature in range(n_features):
            parent = int(model["parents"][feature])
            if parent == -1:
                scores[row_index] += model["log_probabilities"][feature][:, indices[feature]]
            else:
                scores[row_index] += model["log_probabilities"][feature][:, indices[parent], indices[feature]]
    return scores


def predict_tan(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    scores = tan_log_scores(model, X)
    return model["classes"][np.argmax(scores, axis=1)]
