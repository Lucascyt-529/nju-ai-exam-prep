"""学生练习：TAN条件互信息、最大生成树与分类。"""

import numpy as np


def conditional_mutual_information(X: np.ndarray, y: np.ndarray,
                                   feature_i: int, feature_j: int) -> float:
    raise NotImplementedError("请完成 conditional_mutual_information")


def conditional_mi_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 conditional_mi_matrix")


def maximum_spanning_tree(weights: np.ndarray, *, root: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 maximum_spanning_tree")


def fit_tan(X: np.ndarray, y: np.ndarray, *, root: int = 0,
            alpha: float = 1.0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_tan")


def tan_log_scores(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 tan_log_scores")


def predict_tan(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 predict_tan")
