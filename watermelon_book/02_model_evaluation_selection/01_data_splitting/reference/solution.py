"""参考实现：留出法、分层划分、K折与自助采样。"""

import numpy as np


def _validate_n_samples(n_samples: int) -> None:
    if not isinstance(n_samples, int):
        raise TypeError("n_samples 必须是整数")
    if n_samples < 2:
        raise ValueError("至少需要2个样本")


def _validate_test_size(test_size: float) -> None:
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size 必须位于 (0, 1)")


def _validate_labels(y: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size < 2:
        raise ValueError("y 必须是至少含2个样本的一维数组")


def train_test_split_indices(
    n_samples: int, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """返回互不重叠的训练和测试索引。"""
    _validate_n_samples(n_samples)
    _validate_test_size(test_size)
    n_test = int(np.ceil(n_samples * test_size))
    if n_test >= n_samples:
        raise ValueError("test_size 导致训练集为空")
    permutation = np.random.default_rng(seed).permutation(n_samples)
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]
    return train_indices, test_indices


def stratified_train_test_split_indices(
    y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """每个类别分别划分，再合并训练和测试索引。"""
    _validate_labels(y)
    _validate_test_size(test_size)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if class_indices.size < 2:
            raise ValueError(f"类别 {label!r} 至少需要2个样本才能分层留出")
        shuffled = rng.permutation(class_indices)
        n_test = int(np.ceil(class_indices.size * test_size))
        n_test = min(max(1, n_test), class_indices.size - 1)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return train_indices, test_indices


def kfold_indices(
    n_samples: int, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回K组训练索引和验证索引。"""
    _validate_n_samples(n_samples)
    if not isinstance(n_splits, int) or not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits 必须是2到样本数之间的整数")
    shuffled = np.random.default_rng(seed).permutation(n_samples)
    validation_folds = np.array_split(shuffled, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for validation_indices in validation_folds:
        train_indices = np.setdiff1d(shuffled, validation_indices, assume_unique=True)
        folds.append((train_indices, validation_indices.copy()))
    return folds


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回尽量保持类别比例的K折索引。"""
    _validate_labels(y)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits 必须是至少为2的整数")
    rng = np.random.default_rng(seed)
    validation_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if class_indices.size < n_splits:
            raise ValueError(f"类别 {label!r} 的样本数少于折数")
        shuffled = rng.permutation(class_indices)
        for fold_index, part in enumerate(np.array_split(shuffled, n_splits)):
            validation_parts[fold_index].append(part)

    all_indices = np.arange(y.size)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for parts in validation_parts:
        validation_indices = np.sort(np.concatenate(parts))
        train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        folds.append((train_indices, validation_indices))
    return folds


def bootstrap_indices(
    n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """返回等量有放回训练索引和未抽中的袋外索引。"""
    _validate_n_samples(n_samples)
    training_indices = np.random.default_rng(seed).integers(
        0, n_samples, size=n_samples
    )
    out_of_bag = np.setdiff1d(np.arange(n_samples), np.unique(training_indices))
    return training_indices, out_of_bag
