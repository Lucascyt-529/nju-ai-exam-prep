"""学生练习：随机特征子空间树桩森林。"""

import numpy as np


def sample_feature_subsets(n_features: int, n_estimators: int, max_features: int, *, random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 sample_feature_subsets")


def fit_random_subspace_forest(X: np.ndarray, y: np.ndarray, *, n_estimators: int = 20, max_features: int = 1, random_state: int = 0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_random_subspace_forest")


def base_predictions(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 base_predictions")
