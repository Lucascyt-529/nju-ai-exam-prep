"""参考实现：Bootstrap树桩Bagging与袋外预测。"""

import numpy as np


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def _matrix(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须是非空有限数值二维数组")


def _labels(y: np.ndarray, n: int) -> None:
    if not isinstance(y, np.ndarray) or y.shape != (n,) or set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError("y必须是形状(n,)且同时包含-1和+1")


def bootstrap_sample_indices(n_samples: int, n_estimators: int, *, random_state: int = 0) -> np.ndarray:
    n, B = _positive_int(n_samples, "n_samples"), _positive_int(n_estimators, "n_estimators")
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state必须是整数")
    return np.random.default_rng(int(random_state)).integers(0, n, size=(B, n))


def out_of_bag_indices(indices: np.ndarray, n_samples: int) -> np.ndarray:
    n = _positive_int(n_samples, "n_samples")
    if not isinstance(indices, np.ndarray) or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices必须是一维合法整数下标")
    in_bag = np.zeros(n, dtype=bool); in_bag[np.unique(indices)] = True
    return np.flatnonzero(~in_bag)


def _thresholds(column: np.ndarray) -> np.ndarray:
    values = np.unique(column)
    return np.concatenate(([np.nextafter(values[0], -np.inf)], (values[:-1] + values[1:]) / 2, [np.nextafter(values[-1], np.inf)]))


def _stump_predict(learner: dict[str, object], X: np.ndarray) -> np.ndarray:
    if learner["kind"] == "constant":
        return np.full(X.shape[0], learner["label"], dtype=int)
    base = np.where(X[:, learner["feature"]] >= learner["threshold"], 1, -1)
    return base * learner["polarity"]


def _fit_stump(X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    classes, counts = np.unique(y, return_counts=True)
    if classes.size == 1:
        return {"kind": "constant", "label": int(classes[0])}
    best = None
    for feature in range(X.shape[1]):
        for threshold in _thresholds(X[:, feature]):
            for order, polarity in enumerate((1, -1)):
                base = np.where(X[:, feature] >= threshold, 1, -1) * polarity
                error = int(np.count_nonzero(base != y))
                key = (error, feature, float(threshold), order)
                if best is None or key < best[0]:
                    best = (key, {"kind": "stump", "feature": feature, "threshold": float(threshold), "polarity": polarity})
    return best[1]


def fit_bagging_stumps(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20, random_state: int = 0) -> dict[str, object]:
    _matrix(X); _labels(y, X.shape[0])
    indices = bootstrap_sample_indices(X.shape[0], n_estimators, random_state=random_state)
    learners = tuple(_fit_stump(X[row], y[row]) for row in indices)
    return {"learners": learners, "bootstrap_indices": indices, "n_features": X.shape[1], "n_train": X.shape[0]}


def base_predictions(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _matrix(X)
    if not isinstance(model, dict) or set(model) != {"learners", "bootstrap_indices", "n_features", "n_train"} or X.shape[1] != model["n_features"]:
        raise ValueError("model无效或特征数不匹配")
    return np.vstack([_stump_predict(learner, X) for learner in model["learners"]])


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.mean(base_predictions(model, X), axis=0)


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.where(decision_function(model, X) >= 0, 1, -1)


def oob_decision_function(model: dict[str, object], X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _matrix(X_train)
    if X_train.shape != (model.get("n_train"), model.get("n_features")):
        raise ValueError("X_train形状必须与建模训练集一致")
    predictions = base_predictions(model, X_train)
    sums = np.zeros(X_train.shape[0]); counts = np.zeros(X_train.shape[0], dtype=int)
    for learner_index, indices in enumerate(model["bootstrap_indices"]):
        oob = out_of_bag_indices(indices, X_train.shape[0])
        sums[oob] += predictions[learner_index, oob]; counts[oob] += 1
    scores = np.full(X_train.shape[0], np.nan); covered = counts > 0
    scores[covered] = sums[covered] / counts[covered]
    return scores, counts


def oob_accuracy(model: dict[str, object], X_train: np.ndarray, y_train: np.ndarray) -> tuple[float, np.ndarray]:
    _labels(y_train, X_train.shape[0])
    scores, counts = oob_decision_function(model, X_train); covered = counts > 0
    if not np.any(covered):
        raise ValueError("没有任何样本获得OOB覆盖")
    prediction = np.where(scores[covered] >= 0, 1, -1)
    return float(np.mean(prediction == y_train[covered])), covered
