"""参考实现：Bootstrap加随机候选特征的树桩森林。"""

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


def sample_feature_subsets(n_features: int, n_estimators: int, max_features: int, *, random_state: int = 0) -> np.ndarray:
    d, B, m = _positive_int(n_features, "n_features"), _positive_int(n_estimators, "n_estimators"), _positive_int(max_features, "max_features")
    if m > d:
        raise ValueError("max_features不能超过n_features")
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state必须是整数")
    rng = np.random.default_rng(int(random_state))
    return np.vstack([np.sort(rng.choice(d, size=m, replace=False)) for _ in range(B)])


def _thresholds(column: np.ndarray) -> np.ndarray:
    values = np.unique(column)
    return np.concatenate(([np.nextafter(values[0], -np.inf)], (values[:-1]+values[1:])/2, [np.nextafter(values[-1], np.inf)]))


def _fit_restricted_stump(X: np.ndarray, y: np.ndarray, features: np.ndarray) -> dict[str, object]:
    classes = np.unique(y)
    if classes.size == 1:
        return {"kind": "constant", "label": int(classes[0])}
    best = None
    for feature in features:
        for threshold in _thresholds(X[:, feature]):
            for order, polarity in enumerate((1, -1)):
                prediction = np.where(X[:, feature] >= threshold, 1, -1) * polarity
                key = (int(np.count_nonzero(prediction != y)), int(feature), float(threshold), order)
                if best is None or key < best[0]:
                    best = (key, {"kind": "stump", "feature": int(feature), "threshold": float(threshold), "polarity": polarity})
    return best[1]


def _predict_learner(learner: dict[str, object], X: np.ndarray) -> np.ndarray:
    if learner["kind"] == "constant": return np.full(X.shape[0], learner["label"], dtype=int)
    return np.where(X[:, learner["feature"]] >= learner["threshold"], 1, -1) * learner["polarity"]


def fit_random_subspace_forest(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20, max_features: int = 1, random_state: int = 0) -> dict[str, object]:
    _matrix(X); _labels(y, X.shape[0])
    B = _positive_int(n_estimators, "n_estimators")
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state必须是整数")
    m = _positive_int(max_features, "max_features")
    if m > X.shape[1]:
        raise ValueError("max_features不能超过特征数")
    rng = np.random.default_rng(int(random_state))
    bootstrap = rng.integers(0, X.shape[0], size=(B, X.shape[0]))
    feature_subsets = np.vstack([np.sort(rng.choice(X.shape[1], size=m, replace=False)) for _ in range(B)])
    learners = tuple(_fit_restricted_stump(X[row], y[row], features) for row, features in zip(bootstrap, feature_subsets))
    return {"learners": learners, "bootstrap_indices": bootstrap, "feature_subsets": feature_subsets,
            "n_features": X.shape[1], "max_features": m}


def base_predictions(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _matrix(X)
    if not isinstance(model, dict) or set(model) != {"learners", "bootstrap_indices", "feature_subsets", "n_features", "max_features"} or X.shape[1] != model["n_features"]:
        raise ValueError("model无效或特征数不匹配")
    return np.vstack([_predict_learner(learner, X) for learner in model["learners"]])


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.mean(base_predictions(model, X), axis=0)


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.where(decision_function(model, X) >= 0, 1, -1)


def mean_pairwise_prediction_correlation(predictions: np.ndarray) -> float:
    if not isinstance(predictions, np.ndarray) or predictions.ndim != 2 or predictions.shape[0] < 2 or predictions.shape[1] == 0 or not np.issubdtype(predictions.dtype, np.number) or not np.all(np.isfinite(predictions)):
        raise ValueError("predictions必须至少含两个学习器的有限二维数组")
    centered = predictions.astype(float) - np.mean(predictions, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1); values = []
    for i in range(predictions.shape[0]-1):
        for j in range(i+1, predictions.shape[0]):
            if norms[i] > 0 and norms[j] > 0:
                values.append(float(centered[i] @ centered[j] / (norms[i] * norms[j])))
    return float(np.mean(values)) if values else float("nan")
