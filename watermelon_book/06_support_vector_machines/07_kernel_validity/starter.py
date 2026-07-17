"""学生练习：核矩阵半正定判断、有限特征坐标与核组合。"""

from collections.abc import Sequence

import numpy as np


def kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    kernel: str = "linear",
    gamma: float = 1.0,
    degree: int = 2,
    coef0: float = 0.0,
) -> np.ndarray:
    raise NotImplementedError("请完成 kernel_matrix")


def gram_diagnostics(
    gram: np.ndarray,
    *,
    symmetry_tolerance: float = 1e-10,
    eigen_tolerance: float = 1e-10,
) -> dict[str, object]:
    raise NotImplementedError("请完成 gram_diagnostics")


def quadratic_form(gram: np.ndarray, coefficients: np.ndarray) -> float:
    raise NotImplementedError("请完成 quadratic_form")


def finite_feature_coordinates(
    gram: np.ndarray, *, tolerance: float = 1e-10
) -> np.ndarray:
    raise NotImplementedError("请完成 finite_feature_coordinates")


def positive_weighted_sum(
    grams: Sequence[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 positive_weighted_sum")


def product_kernel_grams(grams: Sequence[np.ndarray]) -> np.ndarray:
    raise NotImplementedError("请完成 product_kernel_grams")


def scale_kernel_gram(gram: np.ndarray, values: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 scale_kernel_gram")
