"""参考实现：用折外预测构造Stacking次级数据，并训练线性组合器。"""

from __future__ import annotations

from numbers import Real
from typing import Callable

import numpy as np


def _features_targets(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError("X必须是非空二维数组")
    if not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X)):
        raise ValueError("X必须是有限数值数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y必须是与X样本数一致的一维数组")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须是有限数值数组")
    return X.astype(float, copy=False), y.astype(float, copy=False)


def _random_state(random_state: int) -> int:
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state必须是整数")
    return int(random_state)


def kfold_validation_indices(n_samples: int, n_splits: int, *,
                             shuffle: bool = True,
                             random_state: int = 0) -> tuple[np.ndarray, ...]:
    """返回互不重叠且恰好覆盖全部样本的K份验证下标。"""
    if (isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer))
            or n_samples <= 0):
        raise ValueError("n_samples必须是正整数")
    if (isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, (int, np.integer))
            or not 2 <= n_splits <= n_samples):
        raise ValueError("n_splits必须满足2 <= n_splits <= n_samples")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError("shuffle必须是布尔值")
    seed = _random_state(random_state)

    indices = np.arange(int(n_samples))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    fold_sizes = np.full(int(n_splits), int(n_samples) // int(n_splits), dtype=int)
    fold_sizes[: int(n_samples) % int(n_splits)] += 1
    boundaries = np.concatenate(([0], np.cumsum(fold_sizes)))
    return tuple(np.sort(indices[boundaries[i]:boundaries[i + 1]]) for i in range(int(n_splits)))


def _fit_new_model(factory: Callable[[], object], X: np.ndarray, y: np.ndarray) -> object:
    if not callable(factory):
        raise ValueError("每个base_factory都必须可调用")
    model = factory()
    if not hasattr(model, "fit") or not callable(model.fit):
        raise ValueError("基学习器必须提供fit(X, y)")
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ValueError("基学习器必须提供predict(X)")
    fitted = model.fit(X, y)
    fitted_model = model if fitted is None else fitted
    if not hasattr(fitted_model, "predict") or not callable(fitted_model.predict):
        raise ValueError("fit返回的基学习器必须提供predict(X)")
    return fitted_model


def _predict_vector(model: object, X: np.ndarray, expected_size: int) -> np.ndarray:
    prediction = np.asarray(model.predict(X))
    if prediction.shape != (expected_size,):
        raise ValueError("每个基学习器必须对每个样本输出一个数，形状为(n,)")
    if not np.issubdtype(prediction.dtype, np.number) or not np.all(np.isfinite(prediction)):
        raise ValueError("基学习器预测必须是有限数值")
    return prediction.astype(float, copy=False)


def build_oof_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    base_factories: list[Callable[[], object]] | tuple[Callable[[], object], ...],
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
) -> dict[str, object]:
    """构造折外次级特征，并另用全部训练数据拟合部署时的基模型。"""
    X_float, y_float = _features_targets(X, y)
    try:
        factories = tuple(base_factories)
    except TypeError as exc:
        raise ValueError("base_factories必须是非空工厂序列") from exc
    if not factories:
        raise ValueError("base_factories不能为空")

    folds = kfold_validation_indices(
        len(X_float), n_splits, shuffle=shuffle, random_state=random_state
    )
    all_indices = np.arange(len(X_float))
    meta = np.empty((len(X_float), len(factories)), dtype=float)
    final_models: list[object] = []

    for column, factory in enumerate(factories):
        for validation_indices in folds:
            train_mask = np.ones(len(X_float), dtype=bool)
            train_mask[validation_indices] = False
            train_indices = all_indices[train_mask]
            model = _fit_new_model(factory, X_float[train_indices], y_float[train_indices])
            meta[validation_indices, column] = _predict_vector(
                model, X_float[validation_indices], len(validation_indices)
            )
        final_models.append(_fit_new_model(factory, X_float, y_float))

    return {
        "meta_features": meta,
        "base_models": tuple(final_models),
        "validation_folds": folds,
    }


def fit_ridge_combiner(meta_features: np.ndarray, y: np.ndarray, *,
                       l2: float = 1e-8) -> dict[str, np.ndarray | float]:
    """拟合不惩罚截距的岭回归次级学习器。"""
    if (not isinstance(meta_features, np.ndarray) or meta_features.ndim != 2
            or 0 in meta_features.shape or not np.issubdtype(meta_features.dtype, np.number)
            or not np.all(np.isfinite(meta_features))):
        raise ValueError("meta_features必须是非空有限数值二维数组")
    if (not isinstance(y, np.ndarray) or y.shape != (meta_features.shape[0],)
            or not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y))):
        raise ValueError("y必须是与次级样本数一致的有限数值一维数组")
    if (isinstance(l2, (bool, np.bool_)) or not isinstance(l2, Real)
            or not np.isfinite(l2) or l2 < 0):
        raise ValueError("l2必须是非负有限实数")

    design = np.column_stack((np.ones(len(meta_features)), meta_features.astype(float)))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    parameters = np.linalg.pinv(design.T @ design + float(l2) * penalty) @ design.T @ y.astype(float)
    return {"intercept": float(parameters[0]), "weights": parameters[1:]}


def predict_ridge_combiner(meta_features: np.ndarray,
                           combiner: dict[str, np.ndarray | float]) -> np.ndarray:
    if (not isinstance(meta_features, np.ndarray) or meta_features.ndim != 2
            or not np.issubdtype(meta_features.dtype, np.number)
            or not np.all(np.isfinite(meta_features))):
        raise ValueError("meta_features必须是有限数值二维数组")
    if not isinstance(combiner, dict) or set(combiner) != {"intercept", "weights"}:
        raise ValueError("combiner格式错误")
    weights = np.asarray(combiner["weights"])
    intercept = combiner["intercept"]
    if (weights.shape != (meta_features.shape[1],) or not np.all(np.isfinite(weights))
            or not isinstance(intercept, Real) or not np.isfinite(intercept)):
        raise ValueError("combiner参数与次级特征不匹配")
    return float(intercept) + meta_features.astype(float) @ weights.astype(float)


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
    """完成OOF次级数据、次级学习器和全数据基模型三部分训练。"""
    oof = build_oof_meta_features(
        X, y, base_factories, n_splits=n_splits,
        shuffle=shuffle, random_state=random_state,
    )
    combiner = fit_ridge_combiner(oof["meta_features"], y, l2=l2)
    return {
        "base_models": oof["base_models"],
        "combiner": combiner,
        "oof_meta_features": oof["meta_features"],
        "validation_folds": oof["validation_folds"],
    }


def predict_stacking_regressor(X: np.ndarray, model: dict[str, object]) -> np.ndarray:
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X))):
        raise ValueError("X必须是有限数值二维数组")
    if not isinstance(model, dict) or not {"base_models", "combiner"}.issubset(model):
        raise ValueError("model格式错误")
    base_models = tuple(model["base_models"])
    if not base_models:
        raise ValueError("model必须包含至少一个基模型")
    meta = np.column_stack([
        _predict_vector(base_model, X.astype(float, copy=False), len(X))
        for base_model in base_models
    ])
    return predict_ridge_combiner(meta, model["combiner"])
