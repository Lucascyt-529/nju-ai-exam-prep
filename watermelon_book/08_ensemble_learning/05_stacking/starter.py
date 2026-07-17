"""学生练习：用折外预测完成Stacking回归。"""

from typing import Callable

import numpy as np


def kfold_validation_indices(n_samples: int, n_splits: int, *,
                             shuffle: bool = True,
                             random_state: int = 0) -> tuple[np.ndarray, ...]:
    raise NotImplementedError("请完成 kfold_validation_indices")


def build_oof_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    base_factories: list[Callable[[], object]] | tuple[Callable[[], object], ...],
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> dict[str, object]:
    raise NotImplementedError("请完成 build_oof_meta_features")


def fit_ridge_combiner(meta_features: np.ndarray, y: np.ndarray, *,
                       l2: float = 1e-8) -> dict[str, np.ndarray | float]:
    raise NotImplementedError("请完成 fit_ridge_combiner")


def predict_ridge_combiner(meta_features: np.ndarray,
                           combiner: dict[str, np.ndarray | float]) -> np.ndarray:
    raise NotImplementedError("请完成 predict_ridge_combiner")


def fit_stacking_regressor(
    X: np.ndarray,
    y: np.ndarray,
    base_factories: list[Callable[[], object]] | tuple[Callable[[], object], ...],
    *,
    n_splits: int = 5,
    l2: float = 1e-8,
    shuffle: bool = True,
    random_state: int = 0,
) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_stacking_regressor")


def predict_stacking_regressor(X: np.ndarray, model: dict[str, object]) -> np.ndarray:
    raise NotImplementedError("请完成 predict_stacking_regressor")
