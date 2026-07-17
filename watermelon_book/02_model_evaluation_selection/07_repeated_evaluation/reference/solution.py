"""参考实现：重复留出与重复交叉验证。"""

from collections.abc import Callable

import numpy as np


Fold = tuple[np.ndarray, np.ndarray]


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} 必须是正整数")
    if int(value) < 1:
        raise ValueError(f"{name} 必须是正整数")


def _validate_supervised_data(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError("X 必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y 必须是一维数组且样本数与X一致")
    if X.shape[0] < 2:
        raise ValueError("至少需要2个样本")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError("数值标签必须只包含有限值")


def _validate_prediction(prediction: np.ndarray, expected_size: int) -> np.ndarray:
    values = np.asarray(prediction)
    if values.ndim != 1 or values.shape[0] != expected_size:
        raise ValueError("predict 必须为每个评估样本返回一个一维预测")
    if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
        raise ValueError("数值预测必须只包含有限值")
    return values


def _score_once(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    model = fit_model(X[train_indices], y[train_indices])
    prediction = _validate_prediction(
        predict(model, X[test_indices]), test_indices.size
    )
    score = float(metric(y[test_indices], prediction))
    if not np.isfinite(score):
        raise ValueError("metric 必须返回有限标量")
    return score


def generate_repetition_seeds(repetitions: int, base_seed: int) -> np.ndarray:
    """从基础种子生成每次重复使用的独立、可复现子种子。"""
    _validate_positive_integer(repetitions, "repetitions")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed 必须是非负整数")
    if int(base_seed) < 0:
        raise ValueError("base_seed 必须是非负整数")

    children = np.random.SeedSequence(int(base_seed)).spawn(int(repetitions))
    return np.asarray(
        [child.generate_state(1, dtype=np.uint32)[0] for child in children],
        dtype=np.uint64,
    )


def _holdout_indices(
    y: np.ndarray, test_size: float, seed: int, *, stratified: bool
) -> Fold:
    if isinstance(test_size, bool) or not np.isscalar(test_size):
        raise TypeError("test_size 必须是位于(0, 1)的数")
    if not np.isfinite(test_size) or not 0 < float(test_size) < 1:
        raise ValueError("test_size 必须位于(0, 1)")
    if not isinstance(stratified, bool):
        raise TypeError("stratified 必须是布尔值")

    rng = np.random.default_rng(seed)
    if not stratified:
        n_test = int(np.ceil(y.size * float(test_size)))
        if n_test >= y.size:
            raise ValueError("test_size 导致训练集为空")
        shuffled = rng.permutation(y.size)
        return shuffled[n_test:], shuffled[:n_test]

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if class_indices.size < 2:
            raise ValueError(f"类别 {label!r} 至少需要2个样本才能分层留出")
        shuffled = rng.permutation(class_indices)
        n_test = int(np.ceil(class_indices.size * float(test_size)))
        n_test = min(max(1, n_test), class_indices.size - 1)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    return (
        rng.permutation(np.concatenate(train_parts)),
        rng.permutation(np.concatenate(test_parts)),
    )


def _kfold_indices(
    y: np.ndarray, n_splits: int, seed: int, *, stratified: bool
) -> list[Fold]:
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits 必须是整数")
    if not 2 <= int(n_splits) <= y.size:
        raise ValueError("n_splits 必须是2到样本数之间的整数")
    if not isinstance(stratified, bool):
        raise TypeError("stratified 必须是布尔值")

    rng = np.random.default_rng(seed)
    if not stratified:
        shuffled = rng.permutation(y.size)
        validation_folds = np.array_split(shuffled, int(n_splits))
    else:
        validation_parts: list[list[np.ndarray]] = [
            [] for _ in range(int(n_splits))
        ]
        for label in np.unique(y):
            class_indices = np.flatnonzero(y == label)
            if class_indices.size < int(n_splits):
                raise ValueError(f"类别 {label!r} 的样本数少于折数")
            shuffled_class = rng.permutation(class_indices)
            for fold_index, part in enumerate(
                np.array_split(shuffled_class, int(n_splits))
            ):
                validation_parts[fold_index].append(part)
        validation_folds = [
            rng.permutation(np.concatenate(parts)) for parts in validation_parts
        ]

    all_indices = np.arange(y.size)
    return [
        (
            np.setdiff1d(all_indices, validation_indices, assume_unique=True),
            np.asarray(validation_indices, dtype=int).copy(),
        )
        for validation_indices in validation_folds
    ]


def repeated_holdout_scores(
    X: np.ndarray,
    y: np.ndarray,
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repetitions: int,
    test_size: float,
    base_seed: int,
    stratified: bool = False,
) -> dict[str, object]:
    """重复随机留出并返回分数、种子及每次训练/测试索引。"""
    _validate_supervised_data(X, y)
    seeds = generate_repetition_seeds(repetitions, base_seed)
    scores = np.empty(int(repetitions), dtype=float)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for repeat_index, seed in enumerate(seeds):
        train_indices, test_indices = _holdout_indices(
            y, test_size, int(seed), stratified=stratified
        )
        scores[repeat_index] = _score_once(
            X,
            y,
            train_indices,
            test_indices,
            fit_model,
            predict,
            metric,
        )
        train_parts.append(train_indices.copy())
        test_parts.append(test_indices.copy())

    return {
        "scores": scores,
        "seeds": seeds,
        "train_indices": tuple(train_parts),
        "test_indices": tuple(test_parts),
    }


def repeated_kfold_scores(
    X: np.ndarray,
    y: np.ndarray,
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repetitions: int,
    n_splits: int,
    base_seed: int,
    stratified: bool = False,
) -> dict[str, object]:
    """重复K折评估并返回(p, k)分数矩阵、种子及各套折索引。"""
    _validate_supervised_data(X, y)
    seeds = generate_repetition_seeds(repetitions, base_seed)
    scores = np.empty((int(repetitions), int(n_splits)), dtype=float)
    repeated_folds: list[tuple[Fold, ...]] = []

    for repeat_index, seed in enumerate(seeds):
        folds = _kfold_indices(
            y, n_splits, int(seed), stratified=stratified
        )
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            scores[repeat_index, fold_index] = _score_once(
                X,
                y,
                train_indices,
                test_indices,
                fit_model,
                predict,
                metric,
            )
        repeated_folds.append(
            tuple((train.copy(), test.copy()) for train, test in folds)
        )

    return {"scores": scores, "seeds": seeds, "folds": tuple(repeated_folds)}


def summarize_repeated_scores(scores: np.ndarray) -> dict[str, object]:
    """汇总运行均值、运行波动、每次重复均值及实际拟合次数。"""
    values = np.asarray(scores, dtype=float)
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("scores 必须是非空的一维或二维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores 必须只包含有限值")

    repeat_means = values.copy() if values.ndim == 1 else values.mean(axis=1)
    return {
        "overall_mean": float(values.mean()),
        "run_std": float(values.std()),
        "repeat_means": repeat_means,
        "repeat_mean_std": float(repeat_means.std()),
        "fit_count": int(values.size),
    }


def evaluation_run_count(
    repetitions: int, *, n_splits: int | None = None
) -> int:
    """返回重复留出或重复K折所需的模型拟合次数。"""
    _validate_positive_integer(repetitions, "repetitions")
    if n_splits is None:
        return int(repetitions)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits 必须是至少为2的整数")
    if int(n_splits) < 2:
        raise ValueError("n_splits 必须是至少为2的整数")
    return int(repetitions) * int(n_splits)
