"""参考实现：确定性决策树桩与二分类AdaBoost。"""

import numpy as np


def _matrix(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须是非空有限数值二维数组")


def _labels(y: np.ndarray, n: int) -> None:
    if not isinstance(y, np.ndarray) or y.shape != (n,) or set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError("y必须是形状(n,)且同时包含-1和+1")


def _weights(sample_weight: np.ndarray, n: int) -> None:
    if not isinstance(sample_weight, np.ndarray) or sample_weight.shape != (n,) or not np.issubdtype(sample_weight.dtype, np.number) or not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0) or not np.isclose(np.sum(sample_weight), 1.0):
        raise ValueError("sample_weight必须是形状(n,)且和为1的非负有限数组")


def stump_predict(X: np.ndarray, feature: int, threshold: float, polarity: int) -> np.ndarray:
    _matrix(X)
    if not isinstance(feature, (int, np.integer)) or isinstance(feature, (bool, np.bool_)) or feature < 0 or feature >= X.shape[1]:
        raise ValueError("feature下标无效")
    if not isinstance(threshold, (int, float, np.integer, np.floating)) or not np.isfinite(threshold) or polarity not in (-1, 1):
        raise ValueError("threshold或polarity无效")
    base = np.where(X[:, int(feature)] >= float(threshold), 1, -1)
    return base * int(polarity)


def candidate_thresholds(column: np.ndarray) -> np.ndarray:
    if not isinstance(column, np.ndarray) or column.ndim != 1 or column.size == 0 or not np.issubdtype(column.dtype, np.number) or not np.all(np.isfinite(column)):
        raise ValueError("column必须是非空有限数值一维数组")
    values = np.unique(column.astype(float))
    middle = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(([np.nextafter(values[0], -np.inf)], middle, [np.nextafter(values[-1], np.inf)]))


def fit_weighted_stump(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, object]:
    _matrix(X); _labels(y, X.shape[0]); _weights(sample_weight, X.shape[0])
    best = None
    for feature in range(X.shape[1]):
        for threshold in candidate_thresholds(X[:, feature]):
            for polarity_order, polarity in enumerate((1, -1)):
                prediction = stump_predict(X, feature, float(threshold), polarity)
                error = float(np.sum(sample_weight[prediction != y]))
                key = (error, feature, float(threshold), polarity_order)
                if best is None or key < best[0]:
                    best = (key, {"feature": feature, "threshold": float(threshold), "polarity": polarity, "error": error})
    return best[1]


def classifier_weight(error: float, *, epsilon: float = 1e-12) -> float:
    if not isinstance(error, (int, float, np.integer, np.floating)) or isinstance(error, (bool, np.bool_)) or not np.isfinite(error) or error < 0 or error >= 0.5:
        raise ValueError("error必须位于[0,0.5)")
    clipped = max(float(error), float(epsilon))
    return float(0.5 * np.log((1.0 - clipped) / clipped))


def update_sample_weights(sample_weight: np.ndarray, y: np.ndarray, prediction: np.ndarray, alpha: float) -> np.ndarray:
    _weights(sample_weight, sample_weight.size); _labels(y, sample_weight.size)
    if not isinstance(prediction, np.ndarray) or prediction.shape != y.shape or not np.all(np.isin(prediction, [-1, 1])):
        raise ValueError("prediction必须是同形状的-1/+1数组")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha必须是正有限数值")
    updated = sample_weight * np.exp(-float(alpha) * y * prediction)
    return updated / np.sum(updated)


def fit_adaboost(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20) -> dict[str, object]:
    _matrix(X); _labels(y, X.shape[0])
    if not isinstance(n_estimators, (int, np.integer)) or isinstance(n_estimators, (bool, np.bool_)) or n_estimators <= 0:
        raise ValueError("n_estimators必须是正整数")
    weights = np.full(X.shape[0], 1.0 / X.shape[0]); learners, weight_history = [], [weights.copy()]
    for _ in range(int(n_estimators)):
        stump = fit_weighted_stump(X, y, weights); error = stump["error"]
        if error >= 0.5 - 1e-12:
            break
        alpha = classifier_weight(error)
        learner = {**stump, "alpha": alpha}; learners.append(learner)
        prediction = stump_predict(X, stump["feature"], stump["threshold"], stump["polarity"])
        weights = update_sample_weights(weights, y, prediction, alpha); weight_history.append(weights.copy())
        if error <= 1e-12:
            break
    return {"learners": tuple(learners), "n_features": X.shape[1], "weight_history": np.vstack(weight_history)}


def decision_function(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    _matrix(X)
    if not isinstance(model, dict) or set(model) != {"learners", "n_features", "weight_history"} or X.shape[1] != model["n_features"]:
        raise ValueError("model无效或特征数不匹配")
    scores = np.zeros(X.shape[0])
    for learner in model["learners"]:
        scores += learner["alpha"] * stump_predict(X, learner["feature"], learner["threshold"], learner["polarity"])
    return scores


def predict(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    return np.where(decision_function(model, X) >= 0, 1, -1)
