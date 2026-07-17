"""参考实现：直接计数、对数空间的AODE。"""

import numpy as np


def _valid_X(X: np.ndarray, name: str = "X") -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError(f"{name}必须是非空二维数组")
    for value in X.flat:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError(f"{name}不能包含None或非有限值")


def _positive(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def fit_aode(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, min_parent_count: int = 1) -> dict[str, object]:
    _valid_X(X)
    if not isinstance(y, np.ndarray) or y.shape != (X.shape[0],) or np.unique(y).size < 2:
        raise ValueError("y必须是形状(n,)且至少包含两个类别")
    alpha_value = _positive(alpha, "alpha")
    threshold = _positive_int(min_parent_count, "min_parent_count")
    return {"X_train": X.copy(), "y_train": y.copy(), "classes": np.unique(y),
            "categories": tuple(np.unique(X[:, j]) for j in range(X.shape[1])),
            "alpha": alpha_value, "min_parent_count": threshold, "n_features": X.shape[1]}


def _validate_model(model: dict[str, object]) -> None:
    required = {"X_train", "y_train", "classes", "categories", "alpha", "min_parent_count", "n_features"}
    if not isinstance(model, dict) or set(model) != required:
        raise ValueError("model键集合无效")
    _valid_X(model["X_train"], "model X_train")


def eligible_parent_indices(model: dict[str, object], row: np.ndarray) -> np.ndarray:
    _validate_model(model)
    if not isinstance(row, np.ndarray) or row.shape != (model["n_features"],):
        raise ValueError("row必须是形状(d,)的一维样本")
    indices = []
    for feature, value in enumerate(row):
        if np.any(model["categories"][feature] == value) and np.count_nonzero(model["X_train"][:, feature] == value) >= model["min_parent_count"]:
            indices.append(feature)
    return np.array(indices, dtype=int)


def _naive_scores(model: dict[str, object], row: np.ndarray) -> np.ndarray:
    X, y, classes, alpha = model["X_train"], model["y_train"], model["classes"], model["alpha"]
    K, n = classes.size, X.shape[0]
    scores = np.empty(K)
    for k, label in enumerate(classes):
        mask = y == label; count_c = np.count_nonzero(mask)
        score = np.log((count_c + alpha) / (n + alpha * K))
        for j, value in enumerate(row):
            V = model["categories"][j].size + 1
            score += np.log((np.count_nonzero(X[mask, j] == value) + alpha) / (count_c + alpha * V))
        scores[k] = score
    return scores


def naive_joint_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model); _valid_X(X)
    if X.shape[1] != model["n_features"]:
        raise ValueError("X特征数与模型不一致")
    return np.vstack([_naive_scores(model, row) for row in X])


def _spode_scores(model: dict[str, object], row: np.ndarray, parent: int) -> np.ndarray:
    X, y, classes, alpha = model["X_train"], model["y_train"], model["classes"], model["alpha"]
    parent_value = row[parent]
    K, n = classes.size, X.shape[0]
    V_parent = model["categories"][parent].size + 1
    scores = np.empty(K)
    for k, label in enumerate(classes):
        parent_mask = (y == label) & (X[:, parent] == parent_value)
        count_cp = np.count_nonzero(parent_mask)
        score = np.log((count_cp + alpha) / (n + alpha * K * V_parent))
        for child, value in enumerate(row):
            if child == parent:
                continue
            V_child = model["categories"][child].size + 1
            count_child = np.count_nonzero(parent_mask & (X[:, child] == value))
            score += np.log((count_child + alpha) / (count_cp + alpha * V_child))
        scores[k] = score
    return scores


def joint_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _validate_model(model); _valid_X(X)
    if X.shape[1] != model["n_features"]:
        raise ValueError("X特征数与模型不一致")
    output = np.empty((X.shape[0], model["classes"].size))
    for r, row in enumerate(X):
        parents = eligible_parent_indices(model, row)
        if parents.size == 0:
            output[r] = _naive_scores(model, row)
            continue
        component = np.vstack([_spode_scores(model, row, int(parent)) for parent in parents])
        maximum = np.max(component, axis=0)
        output[r] = maximum + np.log(np.mean(np.exp(component - maximum), axis=0))
    return output


def predict_proba(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    scores = joint_log_scores(model, X)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return model["classes"][np.argmax(joint_log_scores(model, X), axis=1)]
