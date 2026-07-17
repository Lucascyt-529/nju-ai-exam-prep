"""参考实现：核矩阵半正定诊断、有限样本特征分解与核函数组合。"""

from collections.abc import Sequence

import numpy as np


KERNELS = {"linear", "polynomial", "rbf", "laplacian", "sigmoid"}


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or not np.issubdtype(value.dtype, np.number)
        or value.ndim != 2
        or 0 in value.shape
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name}必须是非空有限数值二维数组")
    return value.astype(float, copy=False)


def _positive(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def kernel_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    kernel: str = "linear",
    gamma: float = 1.0,
    degree: int = 2,
    coef0: float = 0.0,
) -> np.ndarray:
    """计算查询样本与参照样本的核矩阵，结果形状为(m,n)。"""
    Xf, Zf = _matrix(X, "X"), _matrix(Z, "Z")
    if Xf.shape[1] != Zf.shape[1]:
        raise ValueError("X和Z的特征数必须一致")
    if kernel not in KERNELS:
        raise ValueError("kernel必须是linear、polynomial、rbf、laplacian或sigmoid")
    gamma_value = _positive(gamma, "gamma")
    degree_value = _positive_int(degree, "degree")
    if (
        not isinstance(coef0, (int, float, np.integer, np.floating))
        or isinstance(coef0, (bool, np.bool_))
        or not np.isfinite(coef0)
    ):
        raise ValueError("coef0必须是有限数值")

    dot = Xf @ Zf.T
    if kernel == "linear":
        result = dot
    elif kernel == "polynomial":
        with np.errstate(over="ignore", invalid="ignore"):
            result = (gamma_value * dot + float(coef0)) ** degree_value
    elif kernel in {"rbf", "laplacian"}:
        squared = (
            np.sum(Xf * Xf, axis=1)[:, None]
            + np.sum(Zf * Zf, axis=1)[None, :]
            - 2.0 * dot
        )
        squared = np.maximum(squared, 0.0)
        distance = squared if kernel == "rbf" else np.sqrt(squared)
        result = np.exp(-gamma_value * distance)
    else:
        result = np.tanh(gamma_value * dot + float(coef0))

    if not np.all(np.isfinite(result)):
        raise ValueError("核矩阵出现非有限数值，请调整数据或参数")
    return result


def gram_diagnostics(
    gram: np.ndarray,
    *,
    symmetry_tolerance: float = 1e-10,
    eigen_tolerance: float = 1e-10,
) -> dict[str, object]:
    """诊断一个有限核矩阵是否对称且在容差内半正定。"""
    K = _matrix(gram, "gram")
    if K.shape[0] != K.shape[1]:
        raise ValueError("gram必须是方阵")
    symmetry_tol = _positive(symmetry_tolerance, "symmetry_tolerance")
    eigen_tol = _positive(eigen_tolerance, "eigen_tolerance")
    symmetry_error = float(np.max(np.abs(K - K.T)))
    symmetric = symmetry_error <= symmetry_tol
    symmetric_part = 0.5 * (K + K.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_part)
    minimum = float(eigenvalues[0])
    return {
        "symmetric": symmetric,
        "symmetry_error": symmetry_error,
        "eigenvalues": eigenvalues,
        "minimum_eigenvalue": minimum,
        "positive_semidefinite": bool(symmetric and minimum >= -eigen_tol),
    }


def quadratic_form(gram: np.ndarray, coefficients: np.ndarray) -> float:
    """计算c.T @ K @ c，用于理解半正定定义。"""
    K = _matrix(gram, "gram")
    if K.shape[0] != K.shape[1]:
        raise ValueError("gram必须是方阵")
    if (
        not isinstance(coefficients, np.ndarray)
        or not np.issubdtype(coefficients.dtype, np.number)
        or coefficients.shape != (K.shape[0],)
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("coefficients必须是与gram阶数相同的有限数值一维数组")
    c = coefficients.astype(float, copy=False)
    return float(c @ K @ c)


def finite_feature_coordinates(
    gram: np.ndarray, *, tolerance: float = 1e-10
) -> np.ndarray:
    """为一个PSD有限Gram矩阵构造Phi，使Phi @ Phi.T重建该矩阵。"""
    tol = _positive(tolerance, "tolerance")
    report = gram_diagnostics(
        gram, symmetry_tolerance=tol, eigen_tolerance=tol
    )
    if not report["positive_semidefinite"]:
        raise ValueError("gram不是对称半正定矩阵")
    eigenvalues = report["eigenvalues"]
    K = _matrix(gram, "gram")
    _, eigenvectors = np.linalg.eigh(0.5 * (K + K.T))
    keep = eigenvalues > tol
    if not np.any(keep):
        return np.zeros((K.shape[0], 0), dtype=float)
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]


def _validated_grams(grams: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not isinstance(grams, Sequence) or isinstance(grams, (str, bytes)) or len(grams) == 0:
        raise ValueError("grams必须是非空核矩阵序列")
    checked = [_matrix(gram, f"grams[{index}]") for index, gram in enumerate(grams)]
    shape = checked[0].shape
    if shape[0] != shape[1] or any(gram.shape != shape for gram in checked):
        raise ValueError("所有核矩阵必须是同阶方阵")
    return checked


def positive_weighted_sum(
    grams: Sequence[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """按正权重求核矩阵线性组合。"""
    checked = _validated_grams(grams)
    if (
        not isinstance(weights, np.ndarray)
        or not np.issubdtype(weights.dtype, np.number)
        or weights.shape != (len(checked),)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("weights必须是长度匹配的正有限数值一维数组")
    return np.sum(
        np.stack(checked, axis=0) * weights.astype(float)[:, None, None], axis=0
    )


def product_kernel_grams(grams: Sequence[np.ndarray]) -> np.ndarray:
    """逐元素相乘核矩阵，对应核函数的直积。"""
    checked = _validated_grams(grams)
    result = np.ones_like(checked[0], dtype=float)
    for gram in checked:
        result *= gram
    return result


def scale_kernel_gram(gram: np.ndarray, values: np.ndarray) -> np.ndarray:
    """构造diag(g) @ K @ diag(g)，即g(x)k(x,z)g(z)。"""
    K = _matrix(gram, "gram")
    if K.shape[0] != K.shape[1]:
        raise ValueError("gram必须是方阵")
    if (
        not isinstance(values, np.ndarray)
        or not np.issubdtype(values.dtype, np.number)
        or values.shape != (K.shape[0],)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("values必须是与gram阶数相同的有限数值一维数组")
    g = values.astype(float, copy=False)
    return g[:, None] * K * g[None, :]
