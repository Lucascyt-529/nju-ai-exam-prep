"""参考实现：通用交叉验证、折外预测和候选方案选择。"""

from collections.abc import Callable

import numpy as np


Fold = tuple[np.ndarray, np.ndarray]


def _validate_supervised_data(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y 必须是一维数组且样本数与X一致")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError("数值标签必须只包含有限数值")


def _validated_folds(folds: list[Fold], n_samples: int) -> list[Fold]:
    if not folds:
        raise ValueError("folds 不能为空")
    all_indices = np.arange(n_samples)
    validation_parts: list[np.ndarray] = []
    validated: list[Fold] = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train = np.asarray(train_indices, dtype=int)
        validation = np.asarray(validation_indices, dtype=int)
        if train.ndim != 1 or validation.ndim != 1 or train.size == 0 or validation.size == 0:
            raise ValueError(f"第{fold_number}折的训练和验证索引必须是非空一维数组")
        if np.unique(train).size != train.size or np.unique(validation).size != validation.size:
            raise ValueError(f"第{fold_number}折内部不能有重复索引")
        if np.any(train < 0) or np.any(train >= n_samples) or np.any(validation < 0) or np.any(validation >= n_samples):
            raise IndexError(f"第{fold_number}折包含越界索引")
        if np.intersect1d(train, validation).size:
            raise ValueError(f"第{fold_number}折训练和验证索引重叠")
        expected_train = np.setdiff1d(all_indices, validation, assume_unique=True)
        if not np.array_equal(np.sort(train), expected_train):
            raise ValueError(f"第{fold_number}折训练索引必须是验证索引的完整补集")
        validation_parts.append(validation)
        validated.append((train, validation))

    combined = np.concatenate(validation_parts)
    if combined.size != n_samples or not np.array_equal(np.sort(combined), all_indices):
        raise ValueError("所有验证折必须恰好覆盖每个样本一次")
    return validated


def _validate_predictions(predictions: np.ndarray, expected_size: int) -> np.ndarray:
    result = np.asarray(predictions)
    if result.ndim != 1 or result.shape[0] != expected_size:
        raise ValueError("predict 必须为每个验证样本返回一个一维预测")
    if np.issubdtype(result.dtype, np.number) and not np.all(np.isfinite(result)):
        raise ValueError("数值预测必须只包含有限值")
    return result


def cross_validation_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """返回每个验证折的度量值。"""
    _validate_supervised_data(X, y)
    validated_folds = _validated_folds(folds, X.shape[0])
    scores = np.empty(len(validated_folds), dtype=float)
    for index, (train_indices, validation_indices) in enumerate(validated_folds):
        model = fit_model(X[train_indices], y[train_indices])
        predictions = _validate_predictions(
            predict(model, X[validation_indices]), validation_indices.size
        )
        score = float(metric(y[validation_indices], predictions))
        if not np.isfinite(score):
            raise ValueError("metric 必须返回有限标量")
        scores[index] = score
    return scores


def out_of_fold_predictions(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
) -> np.ndarray:
    """按原始样本顺序返回每个样本恰好一次的折外预测。"""
    _validate_supervised_data(X, y)
    validated_folds = _validated_folds(folds, X.shape[0])
    result = np.empty(X.shape[0], dtype=object)
    for train_indices, validation_indices in validated_folds:
        model = fit_model(X[train_indices], y[train_indices])
        predictions = _validate_predictions(
            predict(model, X[validation_indices]), validation_indices.size
        )
        result[validation_indices] = predictions
    return np.asarray(result.tolist())


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    """返回折分数的均值和总体标准差。"""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores 必须是非空有限一维数组")
    return {"mean": float(values.mean()), "std": float(values.std())}


def select_best_candidate(
    scores_by_name: dict[str, np.ndarray], *, higher_is_better: bool
) -> tuple[str, dict[str, dict[str, float]]]:
    """按平均分选择候选方案，并返回全部摘要。"""
    if not scores_by_name:
        raise ValueError("至少需要一个候选方案")
    summaries = {
        name: summarize_scores(scores) for name, scores in scores_by_name.items()
    }
    means = {name: summary["mean"] for name, summary in summaries.items()}
    best_name = (
        max(means, key=means.get) if higher_is_better else min(means, key=means.get)
    )
    return best_name, summaries
